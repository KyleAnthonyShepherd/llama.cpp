#pragma once

#include "ggml.h"

// Default max tokens per batch for which the tiered path is used.
//
// A cold draw has hot id -1, which every MUL_MAT_ID path skips, so the hot ids hold no duplicates
// and MMQ/MMF serve any batch. 32 stays below the 128-token op-offload threshold: above ~32 tokens
// the tier takes the MoE away from op-offload and pins it to the CPU (PLAN-ngram-mod-expert-cache.md
// sections 1.3 and 16). llama_expert_tier_widen lowers the limit to the MMVQ bound for a weight type
// that has no MMQ path on the device. --expert-tier-max-tokens overrides it.
constexpr int64_t LLAMA_EXPERT_TIER_MAX_TOKENS_DEFAULT = 32;

// current limit, and the setter the context calls at init
void llama_expert_tier_set_max_tokens(int32_t n);

// Expert tier hook: drop-in replacement for ggml_mul_mat_id on expert weight
// tensors that have a registered GPU hot store. Pure stock ggml ops, no
// custom kernel, no per-model code.
//
// How it works (matches oldtricks.md Trick 4 sentinel + LUT design):
//   registered src w (full expert tensor on CPU/mmap) -> {dst_hot (S+1 slots
//   on GPU), per-layer LUTs and masks}. When build_lora_mm_id is called with
//   w, the hook remaps real expert ids through hot_lut/cold_lut, runs two
//   mul_mat_id ops (hot on GPU slots, cold on the full CPU tensor), scales
//   and masks each per routed expert, sums them. Result is byte-identical
//   in shape to a stock mul_mat_id output and feeds straight back into the
//   caller's downstream ops (bias add, activation, second build_lora_mm_id
//   call for down, routing-weight mul).
//
// A cold expert has hot id -1, so the hot mul_mat_id skips it and its rows
// are 0. The cold op computes only the cold experts and leaves 0 at the hot
// ones. Each draw is non-zero in exactly one of the two, so their sum is the
// stock result, before any bias add or activation.

// register one expert weight tensor -> its GPU hot tensor and per-layer LUTs.
// Called by llama_expert_hotstore::allocate() after creating dst_hot and
// the LUT/mask tensors. Multiple entries per layer share the same luts[i].
void llama_expert_tier_register(ggml_tensor * src,
                                ggml_tensor * dst_hot,
                                ggml_tensor * hot_lut,
                                ggml_tensor * cold_mask);

// drop the entire table (called by hotstore destructor)
void llama_expert_tier_clear();

// forget the per-graph cache. Must be called whenever the graph's ggml_context is thrown
// away, because the cached tensors live in it - llm_graph_result::reset() does that.
void llama_expert_tier_reset_graph();

// cheap check: is `w` registered? (used so callers can short-circuit lora)
bool llama_expert_tier_has(ggml_tensor * w);

// drop-in hook called from build_lora_mm_id. Returns the combined
// hot+cold output tensor when `w` is registered; returns nullptr to let the
// caller fall back to stock ggml_mul_mat_id.
//   ctx   : graph context (ctx0 of the calling llm_graph_context)
//   w     : expert weight tensor, ne = [in, out, n_experts], 3d, ne[3]==1
//   cur   : activation, ne = [in, 1, n_tokens], 3d
//   ids   : selected_experts, ne = [n_expert_used, n_tokens], 2d i32, REAL ids
//   w_s   : per-expert quant scale, ne = [n_experts], f32; nullptr if none
ggml_tensor * llama_expert_tier_build(ggml_context * ctx,
                                      ggml_tensor * w,
                                      ggml_tensor * cur,
                                      ggml_tensor * ids,
                                      ggml_tensor * w_s);