#include "llama-expert-tier.h"
#include "llama-ext.h"
#include "llama-impl.h"

#include <atomic>
#include <mutex>
#include <unordered_map>

namespace {
    struct tier_entry {
        ggml_tensor * dst_hot;    // [ne0, ne1, n_expert_used + hot_s]
        ggml_tensor * hot_lut;    // f32 [n_experts]
        ggml_tensor * cold_mask;  // f32 [n_experts] (read as int zero-check by cold op)
        ggml_tensor * draw_pos;   // f32 [n_expert_used], = [0, 1, ... n_expert_used-1]
    };

    std::mutex g_mtx;
    std::unordered_map<ggml_tensor *, tier_entry> g_table;

    std::atomic<int32_t> g_max_tokens{(int32_t) LLAMA_EXPERT_TIER_MAX_TOKENS_DEFAULT};
}

void llama_expert_tier_register(ggml_tensor * src,
                                ggml_tensor * dst_hot,
                                ggml_tensor * hot_lut,
                                ggml_tensor * cold_mask,
                                ggml_tensor * draw_pos) {
    std::lock_guard<std::mutex> lk(g_mtx);
    g_table[src] = {dst_hot, hot_lut, cold_mask, draw_pos};
}

void llama_expert_tier_clear() {
    std::lock_guard<std::mutex> lk(g_mtx);
    g_table.clear();
}

bool llama_expert_tier_has(ggml_tensor * w) {
    std::lock_guard<std::mutex> lk(g_mtx);
    return g_table.find(w) != g_table.end();
}

int32_t llama_expert_tier_max_tokens(void) {
    return g_max_tokens.load(std::memory_order_relaxed);
}

void llama_expert_tier_set_max_tokens(int32_t n) {
    g_max_tokens.store(n > 0 ? n : (int32_t) LLAMA_EXPERT_TIER_MAX_TOKENS_DEFAULT,
            std::memory_order_relaxed);
}

// Build the [n_expert_used, n_tokens] i32 ids the hot path is indexed by.
//
// A cold draw must land on the pad slot for its own draw position j, a hot draw on its expert's
// slot. j is not knowable from a per-expert LUT, so carry it as arithmetic instead:
//
//   hot_lut[e]   = n_expert_used + slot for a hot expert, 0 for a cold one
//   cold_mask[e] = 0 for hot, 1 for cold
//   ids          = hot_lut[e] + cold_mask[e] * j
//
// which gives n_expert_used + slot for hot draws and j for cold ones. Every value is a small
// integer held exactly in f32, so the cast back to i32 is exact. Both gathers use the same
// flatten-then-1d-get_rows pattern as before: it avoids the ggml_repeat_4d + view_2d strides
// that the CUDA mul_mat_id kernel mishandles on multi-token ubatches. `ggml_cont` defends
// against argsort views that may not be contiguous.
static ggml_tensor * remap_ids(ggml_context * ctx,
                              ggml_tensor * lut,
                              ggml_tensor * mask,
                              ggml_tensor * draw_pos,
                              ggml_tensor * selected,
                              int n_experts,
                              int n_expert_used,
                              int n_tokens) {
    ggml_tensor * flat_ids = ggml_reshape_1d(ctx,
        ggml_cont(ctx, selected), n_expert_used * n_tokens);

    ggml_tensor * base = ggml_get_rows(ctx, ggml_reshape_2d(ctx, lut,  1, n_experts), flat_ids);
    ggml_tensor * cold = ggml_get_rows(ctx, ggml_reshape_2d(ctx, mask, 1, n_experts), flat_ids);

    base = ggml_reshape_3d(ctx, base, 1, n_expert_used, n_tokens);
    cold = ggml_reshape_3d(ctx, cold, 1, n_expert_used, n_tokens);

    // a cold expert maps to 0, a real slot whose contribution the caller masks away. Slot 0
    // always exists whenever the store is registered, so this is always in range
    GGML_UNUSED(draw_pos);
    GGML_UNUSED(cold);

    return ggml_cast(ctx, ggml_reshape_2d(ctx, base, n_expert_used, n_tokens), GGML_TYPE_I32);
}

// Build a per-(expert_used, token) mask f32 [1, n_expert_used, n_tokens, 1]
// (broadcastable against a mul_mat_id output of shape [out, n_eu, n_tok, 1]).
// Same flatten pattern as remap_ids.
static ggml_tensor * remap_mask(ggml_context * ctx,
                              ggml_tensor * mask,
                              ggml_tensor * selected,
                              int n_experts,
                              int n_expert_used,
                              int n_tokens) {
    ggml_tensor * mask_rows = ggml_reshape_2d(ctx, mask, 1, n_experts);  // [1, n_experts] f32
    ggml_tensor * flat_ids = ggml_reshape_1d(ctx,
        ggml_cont(ctx, selected), n_expert_used * n_tokens);             // [n_eu*n_tok] i32
    ggml_tensor * r = ggml_get_rows(ctx, mask_rows, flat_ids);           // [1, n_eu*n_tok, 1, 1] f32
    return ggml_reshape_3d(ctx, r, 1, n_expert_used, n_tokens);          // [1, n_eu, n_tok]
}

ggml_tensor * llama_expert_tier_build(ggml_context * ctx,
                                      ggml_tensor * w,
                                      ggml_tensor * cur,
                                      ggml_tensor * ids,
                                      ggml_tensor * w_s) {
    const int64_t n_tokens_max = llama_expert_tier_max_tokens();
    if (cur->ne[2] > n_tokens_max || !ggml_is_quantized(w->type)) {
        // one-shot: a silently bypassed tier looks exactly like an engaged one
        static bool logged = false;
        if (!logged && llama_expert_tier_has(w)) {
            logged = true;
            LLAMA_LOG_WARN("%s: expert tier bypassed: n_tokens=%d (max %d), %s is %s\n",
                    __func__, (int) cur->ne[2], (int) n_tokens_max, w->name, ggml_type_name(w->type));
        }
        return nullptr;
    }

    tier_entry ent;
    {
        std::lock_guard<std::mutex> lk(g_mtx);
        auto it = g_table.find(w);
        if (it == g_table.end()) {
            static bool logged = false;
            if (!logged) {
                logged = true;
                LLAMA_LOG_WARN("%s: expert tier bypassed: %s has no registered hot store\n", __func__, w->name);
            }
            return nullptr;
        }
        ent = it->second;
    }

    const int n_experts     = (int) w->ne[2];
    const int n_expert_used = (int) ids->ne[0];
    const int n_tokens      = (int) cur->ne[2];

    {
        static bool logged = false;
        if (!logged) {
            logged = true;
            LLAMA_LOG_INFO("%s: expert tier engaged: n_tokens=%d, %s\n", __func__, (int) cur->ne[2], w->name);
        }
    }

    // hot path: GPU tier tensor. Remap real expert ids through hot_lut -> hot slot indices.
    ggml_tensor * ids_hot = remap_ids(ctx, ent.hot_lut, ent.cold_mask, ent.draw_pos, ids,
                                      n_experts, n_expert_used, n_tokens);
    ggml_tensor * hot = ggml_mul_mat_id(ctx, ent.dst_hot, cur, ids_hot);

    // A cold draw read a real expert, so drop its contribution here. This is what removes the
    // need for a reserved zero slot: the store holds experts in every slice instead.
    {
        ggml_tensor * cold_d = remap_mask(ctx, ent.cold_mask, ids, n_experts, n_expert_used, n_tokens);
        hot = ggml_mul(ctx, hot, ggml_scale_bias(ctx, cold_d, -1.0f, 1.0f)); // 1 - cold
    }

    // cold path: dedicated CPU op that computes ONLY cold-selected experts.
    // ent.cold_mask is f32 [n_experts] with 1.0f = cold; the op treats it
    // as integer-zero-check (0.0f = hot = skip, non-zero = cold = compute).
    ggml_tensor * cold = ggml_mul_mat_id_cold(ctx, w, cur, ids, ent.cold_mask);

    // per-expert quant scale on both paths
    if (w_s) {
        ggml_tensor * s_rows = ggml_reshape_2d(ctx, w_s, 1, n_experts);
        ggml_tensor * flat_ids = ggml_reshape_1d(ctx,
            ggml_cont(ctx, ids), n_expert_used * n_tokens);
        ggml_tensor * s = ggml_get_rows(ctx, s_rows, flat_ids);
        s = ggml_reshape_3d(ctx, s, 1, n_expert_used, n_tokens);
        hot  = ggml_mul(ctx, hot,  s);
        cold = ggml_mul(ctx, cold, s);
    }

    return ggml_add(ctx, hot, cold);
}