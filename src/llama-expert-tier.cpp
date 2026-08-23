#include "llama-expert-tier.h"
#include "llama-ext.h"
#include "llama-impl.h"

#include <atomic>
#include <mutex>
#include <unordered_map>

namespace {
    struct tier_entry {
        ggml_tensor * dst_hot;    // [ne0, ne1, hot_s]
        ggml_tensor * hot_lut;    // f32 [n_experts]
        ggml_tensor * cold_mask;  // f32 [n_experts] (read as int zero-check by cold op)
    };

    std::mutex g_mtx;
    std::unordered_map<ggml_tensor *, tier_entry> g_table;

    std::atomic<int32_t> g_max_tokens{(int32_t) LLAMA_EXPERT_TIER_MAX_TOKENS_DEFAULT};

    // One layer's gate/up/down are three calls that pass the same router selection, so the
    // flatten and both gathers off it are identical across them. Hold the last one and reuse
    // it. Keyed on the selection AND the layer's LUT so a stale hit is not possible.
    // Only valid inside one graph build: the tensors live in the graph's own context, which
    // llm_graph_result::reset() throws away - hence llama_expert_tier_reset_graph().
    struct tier_memo {
        ggml_tensor * selected = nullptr;
        ggml_tensor * hot_lut  = nullptr;
        ggml_tensor * flat_ids = nullptr;
        ggml_tensor * ids_hot  = nullptr;
        ggml_tensor * hot_d    = nullptr;
    };

    thread_local tier_memo g_memo;
}

void llama_expert_tier_register(ggml_tensor * src,
                                ggml_tensor * dst_hot,
                                ggml_tensor * hot_lut,
                                ggml_tensor * cold_mask) {
    std::lock_guard<std::mutex> lk(g_mtx);
    g_table[src] = {dst_hot, hot_lut, cold_mask};
}

void llama_expert_tier_clear() {
    std::lock_guard<std::mutex> lk(g_mtx);
    g_table.clear();
    g_memo = {};
}

void llama_expert_tier_reset_graph() {
    g_memo = {};
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

// Gather a per-expert f32 LUT into a per-(draw, token) tensor [1, n_expert_used, n_tokens],
// broadcastable against a mul_mat_id output of shape [out, n_eu, n_tok].
// The flatten-then-1d-get_rows pattern avoids the ggml_repeat_4d + view_2d strides that the
// CUDA mul_mat_id kernel mishandles on multi-token ubatches.
static ggml_tensor * gather_per_draw(ggml_context * ctx,
                              ggml_tensor * lut,
                              ggml_tensor * flat_ids,
                              int n_experts,
                              int n_expert_used,
                              int n_tokens) {
    ggml_tensor * r = ggml_get_rows(ctx, ggml_reshape_2d(ctx, lut, 1, n_experts), flat_ids);

    return ggml_reshape_3d(ctx, r, 1, n_expert_used, n_tokens);
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

    // The flatten and the two gathers off it are the same for this layer's gate, up and down,
    // which all pass the same router selection. Build them once and reuse (see tier_memo).
    // ggml_cont defends against argsort views that may not be contiguous.
    if (g_memo.selected != ids || g_memo.hot_lut != ent.hot_lut) {
        ggml_tensor * flat_ids = ggml_reshape_1d(ctx,
            ggml_cont(ctx, ids), n_expert_used * n_tokens);

        // hot_lut[e] = the expert's slot, 0 for a cold one - a real slot whose contribution
        // hot_d removes below. Slot 0 always exists whenever the store is registered, so the
        // index is always in range. Small integers, exact in f32, so the cast back is exact.
        ggml_tensor * base = gather_per_draw(ctx, ent.hot_lut, flat_ids, n_experts, n_expert_used, n_tokens);

        ggml_tensor * cold_d = gather_per_draw(ctx, ent.cold_mask, flat_ids, n_experts, n_expert_used, n_tokens);

        g_memo.selected = ids;
        g_memo.hot_lut  = ent.hot_lut;
        g_memo.flat_ids = flat_ids;
        g_memo.ids_hot  = ggml_cast(ctx, ggml_reshape_2d(ctx, base, n_expert_used, n_tokens), GGML_TYPE_I32);
        g_memo.hot_d    = ggml_scale_bias(ctx, cold_d, -1.0f, 1.0f); // 1 - cold
    }

    // hot path: GPU tier tensor, indexed by hot slot.
    ggml_tensor * hot = ggml_mul_mat_id(ctx, ent.dst_hot, cur, g_memo.ids_hot);

    // cold path: dedicated CPU op that computes ONLY cold-selected experts.
    // ent.cold_mask is f32 [n_experts] with 1.0f = cold; the op treats it
    // as integer-zero-check (0.0f = hot = skip, non-zero = cold = compute).
    ggml_tensor * cold = ggml_mul_mat_id_cold(ctx, w, cur, ids, ent.cold_mask);

    // A cold draw read a real expert, so its contribution has to go. Where there is a
    // per-expert quant scale, ride the mask on that instead of multiplying the matmul output
    // twice: [1, n_eu, n_tok] * [1, n_eu, n_tok] is nothing next to a pass over [n_ff, n_eu, n_tok].
    if (w_s) {
        ggml_tensor * s = gather_per_draw(ctx, w_s, g_memo.flat_ids, n_experts, n_expert_used, n_tokens);

        hot  = ggml_mul(ctx, hot,  ggml_mul(ctx, s, g_memo.hot_d));
        cold = ggml_mul(ctx, cold, s);
    } else {
        hot = ggml_mul(ctx, hot, g_memo.hot_d);
    }

    return ggml_add(ctx, hot, cold);
}