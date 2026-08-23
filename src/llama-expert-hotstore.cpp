#include "llama-expert-hotstore.h"
#include "llama-expert-heatmap.h"
#include "llama-expert-tier.h"
#include "llama-impl.h"
#include "llama-model.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <regex>

// matches the weight tensor of an expert tensor, e.g.:
//   blk.0.ffn_gate_exps.weight
//   blk.3.ffn_down_chexps.weight
// follows the same convention as LLM_FFN_EXPS_REGEX in common.h
static const std::regex g_re_exps_weight("blk\\.(\\d+)\\.ffn_(up|down|gate|gate_up)_(ch|)exps\\.weight");

llama_expert_hotstore::llama_expert_hotstore(
        const llama_model * model, int n_layers, int n_experts, int n_expert_used, int hot_s,
        int sync_period, float hyst, int dwell) :
    n_layers(n_layers),
    n_experts(n_experts),
    hot_s(hot_s),
    hot_s_max(hot_s),
    n_expert_used(n_expert_used),
    bytes_per_slot(n_layers, 0),
    sync_period(sync_period),
    hyst(hyst),
    dwell(dwell) {
    if (n_layers <= 0) {
        return;
    }

    for (const auto & [name, tensor] : llama_internal_get_tensor_map(model)) {
        std::smatch m;
        if (std::regex_search(name, m, g_re_exps_weight)) {
            const int il = std::stoi(m[1].str());
            if (il >= 0 && il < n_layers && tensor->ne[2] > 0) {
                // a slot holds nbytes/n_experts of this tensor
                bytes_per_slot[il] += ggml_nbytes(tensor) / (size_t) tensor->ne[2];
                entries.push_back({il, tensor, nullptr});
            }
        }
    }

    // entries is fixed from here on; build a per-layer index of stable
    // pointers so copy/resync do not iterate the whole entries vector.
    entries_by_layer.assign(n_layers, {});
    for (auto & e : entries) {
        entries_by_layer[e.layer_idx].push_back(&e);
    }

    if (this->hot_s > 0) {
        slot_to_expert.assign(n_layers, std::vector<int>(this->hot_s, -1));
        dwell_count.assign(n_layers, std::vector<int>(this->hot_s, 0));
    }
}

bool llama_expert_hotstore::allocate(ggml_backend_buffer_type_t gpu_buft) {
    if (hot_s <= 0 || entries.empty()) {
        return false;
    }
    if (hot_s > n_experts) {
        throw std::runtime_error(format("%s: hot store S=%d exceeds n_experts=%d",
            __func__, hot_s, n_experts));
    }

    // a no_alloc context just for the hot tensor metadata
    // (also holds the per-layer LUT/mask tensors created below)
    ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * (entries.size() + 4 * n_layers),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ctx = ggml_context_ptr(ggml_init(params));
    if (!ctx) {
        LLAMA_LOG_ERROR("%s: hot store: failed to create ggml context\n", __func__);
        return false;
    }

    // one hot tensor per model expert tensor, hot_s resident slots and nothing reserved -
    // a cold draw's contribution is masked away at the tier output (see the header)
    for (auto & e : entries) {
        e.dst = ggml_new_tensor_3d(ctx.get(), e.src->type, e.src->ne[0], e.src->ne[1], hot_s);
    }

    // per-layer LUTs and masks for in-graph routing (oldtricks Trick 4).
    luts.assign(n_layers, layer_lut{});
    for (int il = 0; il < n_layers; il++) {
        luts[il].hot_lut   = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, n_experts);
        luts[il].cold_mask = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, n_experts);
    }

    // check whether the buffer would fit before committing any VRAM
    const size_t need = ggml_backend_alloc_ctx_tensors_from_buft_size(ctx.get(), gpu_buft);
    if (need == 0) {
        LLAMA_LOG_ERROR("%s: hot store: zero-sized buffer, disabled\n", __func__);
        ctx.reset();
        return false;
    }

    size_t free_mem = 0, total_mem = 0;
    ggml_backend_dev_t dev = ggml_backend_buft_get_device(gpu_buft);
    if (dev) {
        ggml_backend_dev_memory(dev, &free_mem, &total_mem);
    }
    if (dev && free_mem < need) {
        throw std::runtime_error(format("%s: not enough memory to allocate the GPU hot store of %d slots (%zu MiB needed, %zu MiB free on %s)", 
            __func__, hot_s, need / (1024 * 1024), free_mem / (1024 * 1024),
            ggml_backend_dev_name(dev)));
    }

    ggml_backend_buffer_t b = ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), gpu_buft);
    if (b == nullptr) {
        throw std::runtime_error(format("%s: unable to allocate hot store buffer of %d slots (%zu MiB)", 
            __func__, hot_s, need / (1024 * 1024)));
    }
    buf = ggml_backend_buffer_ptr(b);
    ggml_backend_buffer_set_usage(buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    // zero the whole buffer so the pad slots AND every not-yet-filled expert slot are zero.
    // copy_top_s/resync_top_s only write slots at or above n_expert_used, so the pad stays zero
    // for the lifetime of the store.
    ggml_backend_buffer_clear(buf.get(), 0);

    // register each expert weight tensor with the tier hook so build_lora_mm_id
    // can find its GPU hot tensor and per-layer LUTs.
    for (const auto & e : entries) {
        const auto & L = luts[e.layer_idx];
        llama_expert_tier_register(e.src, e.dst, L.hot_lut, L.cold_mask);
    }

    return true;
}

llama_expert_hotstore::~llama_expert_hotstore() {
    llama_expert_tier_clear();
}

void llama_expert_hotstore::copy_top_s(const llama_expert_heatmap & heatmap) {
    if (is_filled || hot_s <= 0 || entries.empty() || !buf) {
        return;
    }

    for (int il = 0; il < n_layers; il++) {
        const std::vector<int> top = heatmap.get_top_s(il, hot_s);
        auto & ste = slot_to_expert[il];
        auto & dc  = dwell_count[il];
        for (int p = 0; p < (int) top.size() && p < hot_s; p++) {
            ste[p] = top[p];
            dc[p]  = dwell; // initial fill is eligible to be corrected next sync
        }

        for (entry * e : entries_by_layer[il]) {
            const size_t slot = ggml_nbytes(e->src) / (size_t) e->src->ne[2];
            const char * src = e->src->data ? (const char *) ggml_get_data(e->src) : nullptr;
            if (!src) {
                continue;
            }
            for (int p = 0; p < hot_s; p++) {
                const int ex = ste[p];
                if (ex < 0) {
                    continue;
                }
                ggml_backend_tensor_set(e->dst, src + (size_t) ex * slot, (size_t) p * slot, slot);
            }
        }
    }

    last_sync_tokens = heatmap.tokens_total;
    is_filled = true;
    update_luts();
    LLAMA_LOG("=== Expert hot store: top-S experts copied to GPU ===\n");
}

size_t llama_expert_hotstore::bytes_per_slot_total() const {
    size_t n = 0;
    for (size_t b : bytes_per_slot) {
        n += b;
    }
    return n;
}

size_t llama_expert_hotstore::bytes_resident() const {
    return buf ? ggml_backend_buffer_get_size(buf.get()) : 0;
}

int llama_expert_hotstore::slots_that_fit(size_t bytes) const {
    const size_t per_slot = bytes_per_slot_total();
    if (per_slot == 0) {
        return 0;
    }

    const int n = (int) (bytes / per_slot);

    return std::max(0, std::min(n, hot_s_max));
}

size_t llama_expert_hotstore::resize(int new_hot_s, const llama_expert_heatmap & heatmap,
                                     ggml_backend_buffer_type_t gpu_buft) {
    new_hot_s = std::min(new_hot_s, hot_s_max);

    if (new_hot_s == hot_s) {
        return 0;
    }

    // a store that never allocated (no CUDA backend, or allocate() failed) stays off
    if (!buf && hot_s > 0) {
        return 0;
    }

    const size_t released = buf ? ggml_backend_buffer_get_size(buf.get()) : 0;

    // the graph holds pointers to dst/LUT tensors, so drop the registrations with them
    llama_expert_tier_clear();
    buf.reset();
    ctx.reset();
    for (auto & e : entries) {
        e.dst = nullptr;
    }
    is_filled = false;

    if (new_hot_s <= 0) {
        hot_s = 0;
        LLAMA_LOG_INFO("%s: expert hot store disabled, released %zu MiB\n", __func__, released / (1024 * 1024));
        return released;
    }

    // allocate() weighs the request against what the device reports free at that instant,
    // which moves under us - compute buffers, fragmentation, a sibling context resizing
    // next. Missing the target must not cost every slot, so step down and try again.
    // The old buffer is already gone by here, so there is nothing to fall back to.
    bool ok = false;

    for (int want = new_hot_s; want > 0 && !ok; ) {
        hot_s = want;
        for (int il = 0; il < n_layers; il++) {
            slot_to_expert[il].assign(hot_s, -1);
            dwell_count[il].assign(hot_s, 0);
        }

        try {
            ok = allocate(gpu_buft);
        } catch (const std::exception & e) {
            LLAMA_LOG_WARN("%s: %s\n", __func__, e.what());
        }

        if (!ok) {
            ctx.reset();

            size_t free_mem = 0, total_mem = 0;
            ggml_backend_dev_t dev = ggml_backend_buft_get_device(gpu_buft);
            if (dev) {
                ggml_backend_dev_memory(dev, &free_mem, &total_mem);
            }

            want = std::min(slots_that_fit(free_mem), want - 1);
        }
    }

    if (!ok) {
        hot_s = 0;
        LLAMA_LOG_WARN("%s: could not re-allocate the hot store at %d slots, cache is now off\n", __func__, new_hot_s);
        return released;
    }
    copy_top_s(heatmap);

    const size_t kept = buf ? ggml_backend_buffer_get_size(buf.get()) : 0;
    LLAMA_LOG_INFO("%s: expert hot store %d slots, %zu MiB\n",
            __func__, hot_s, kept / (1024 * 1024));

    return released > kept ? released - kept : 0;
}

void llama_expert_hotstore::plant_static() {
    if (is_filled || hot_s <= 0 || entries.empty() || !buf) {
        return;
    }

    // plant experts 0..hot_s-1 into slots 0..hot_s-1, one shot, no heatmap
    for (int il = 0; il < n_layers; il++) {
        auto & ste = slot_to_expert[il];
        for (int p = 0; p < hot_s && p < n_experts; p++) {
            ste[p] = p;
        }
        for (entry * e : entries_by_layer[il]) {
            const size_t slot = ggml_nbytes(e->src) / (size_t) e->src->ne[2];
            const char * src = e->src->data ? (const char *) ggml_get_data(e->src) : nullptr;
            if (!src) continue;
            for (int p = 0; p < hot_s && p < n_experts; p++) {
                ggml_backend_tensor_set(e->dst, src + (size_t) p * slot, (size_t) p * slot, slot);
            }
        }
    }

    is_filled = true;
    update_luts();
    LLAMA_LOG("=== Expert hot store: STATIC plant of %d experts per layer ===\n", hot_s);
}

void llama_expert_hotstore::resync_top_s(const llama_expert_heatmap & heatmap) {
    if (!is_filled || hot_s <= 0 || !buf) {
        return;
    }

    // tokens elapsed since the previous sync, used to age dwell counters
    const int64_t elapsed = heatmap.tokens_total - last_sync_tokens;
    int swapped = 0;
    for (int il = 0; il < n_layers; il++) {
        const std::vector<int> top = heatmap.get_top_s(il, hot_s);
        auto & ste = slot_to_expert[il];
        auto & dc  = dwell_count[il];

        // find a free slot index (guard: any resident displacement must
        // clear the hysteresis gate, unless the gate is off)
        auto find_slot = [&](int e_cold) -> int {
            // fill empty slots first (no gate on fill)
            for (int p = 0; p < hot_s; p++) {
                if (ste[p] < 0) {
                    return p;
                }
            }
            if (hyst <= 0.0f) {
                // gate off: displace the weakest resident
                int p_worst = -1;
                for (int p = 0; p < hot_s; p++) {
                    if (ste[p] >= 0 && (p_worst < 0 ||
                        heatmap.get_score(il, ste[p]) < heatmap.get_score(il, ste[p_worst]))) {
                        p_worst = p;
                    }
                }
                return p_worst;
            }
            // gate on: coldest resident that has dwelled enough AND is beaten
            // by hyst * this cold expert
            const float s_cold = heatmap.get_score(il, e_cold);
            int p_worst = -1;
            float worst_score = 1e9f;
            for (int p = 0; p < hot_s; p++) {
                if (ste[p] < 0) {
                    continue;
                }
                if (dc[p] < dwell) {
                    continue; // incumbent must keep its slot (Trick 6)
                }
                if (s_cold >= hyst * heatmap.get_score(il, ste[p])) {
                    const float s_inc = heatmap.get_score(il, ste[p]);
                    if (s_inc < worst_score) {
                        worst_score = s_inc;
                        p_worst     = p;
                    }
                }
            }
            return p_worst;
        };

        // resident experts -> candidate cold experts, most significant first
        std::vector<char> resident_set(n_experts, 0);
        for (int p = 0; p < hot_s; p++) {
            if (ste[p] >= 0) {
                resident_set[ste[p]] = 1;
            }
        }
        for (int e_cold : top) {
            if (e_cold < 0 || e_cold >= n_experts || resident_set[e_cold]) {
                continue;
            }
            const int p = find_slot(e_cold);
            if (p < 0) {
                break; // no slot free or displaceable under the gate
            }
            for (entry * ent : entries_by_layer[il]) {
                const size_t slot = ggml_nbytes(ent->src) / (size_t) ent->src->ne[2];
                const char * src = ent->src->data ? (const char *) ggml_get_data(ent->src) : nullptr;
                if (!src) {
                    continue;
                }
                ggml_backend_tensor_set(ent->dst, src + (size_t) e_cold * slot, (size_t) p * slot, slot);
            }
            ste[p] = e_cold;
            dc[p]  = -elapsed; // fresh dwell: aging below brings it to 0
            swapped++;
        }

        for (int p = 0; p < hot_s; p++) {
            if (ste[p] >= 0) {
                dc[p] += (int) std::max<int64_t>(elapsed, 0);
            }
        }
    }

    last_sync_tokens = heatmap.tokens_total;
    if (swapped > 0) {
        update_luts();
        if (getenv("LLAMA_EXPERT_DEBUG")) {
            LLAMA_LOG("=== Expert hot store: re-sync swapped %d expert slots ===\n", swapped);
        }
    }
}

void llama_expert_hotstore::maybe_resync(const llama_expert_heatmap & heatmap, bool multi_slot) {
    // n_tokens>1 (multi-slot) freezes the hot store: no swapping during the batch
    if (multi_slot || sync_period <= 0 || heatmap.tokens_total <= 0) {
        return;
    }
    if (heatmap.tokens_total / sync_period > last_sync_tokens / sync_period) {
        resync_top_s(heatmap);
    }
}

int llama_expert_hotstore::slot_of(int layer_idx, int expert_id) const {
    if (layer_idx < 0 || layer_idx >= n_layers || hot_s <= 0) {
        return -1;
    }
    const auto & ste = slot_to_expert[layer_idx];
    for (int p = 0; p < hot_s; p++) {
        if (ste[p] == expert_id) {
            return p;
        }
    }
    return -1;
}

void llama_expert_hotstore::update_luts() {
    if (hot_s <= 0 || luts.empty() || !buf) {
        return;
    }

    std::vector<float> hot_lut_h(n_experts);
    std::vector<float> cold_mask_h(n_experts);

    // diagnostic: keep every expert cold, so the whole MoE runs through mul_mat_id_cold and the
    // hot path contributes only zeros. Whatever still differs from stock is then the cold kernel
    // alone, not the hot/cold split. See PLAN-ngram-mod-expert-cache.md section 12.
    const bool cold_only = getenv("LLAMA_EXPERT_TIER_COLD_ONLY") != nullptr;

    for (int il = 0; il < n_layers; il++) {
        const auto & ste = slot_to_expert[il];

        // defaults: everyone cold. 0 is the pad base; the graph adds the draw position to it,
        // so each cold draw of a token ends up on a different pad slot.
        for (int e = 0; e < n_experts; e++) {
            hot_lut_h[e]   = 0.0f;
            cold_mask_h[e] = 1.0f;
        }

        // residents override
        for (int p = 0; p < hot_s && !cold_only; p++) {
            const int e = ste[p];
            if (e < 0) {
                continue;
            }
            hot_lut_h[e]   = (float) p; // its slot
            cold_mask_h[e] = 0.0f;
        }

        const size_t bytes_f32 = n_experts * sizeof(float);
        ggml_backend_tensor_set(luts[il].hot_lut,   hot_lut_h.data(),   0, bytes_f32);
        ggml_backend_tensor_set(luts[il].cold_mask, cold_mask_h.data(), 0, bytes_f32);
    }

    luts_version++;
}

int llama_expert_hotstore::count_cold(const llama_expert_sel & sel) const {
    if (!is_filled || sel.layers.empty()) {
        return -1;
    }

    int cold = 0;
    std::vector<uint8_t> seen(n_experts);

    for (const auto & [il, ids] : sel.layers) {
        std::fill(seen.begin(), seen.end(), 0);
        for (const int32_t id : ids) {
            if (id < 0 || id >= n_experts || seen[id]) {
                continue;
            }
            seen[id] = 1;
            if (slot_of(il, id) < 0) {
                cold++;
            }
        }
    }

    return cold;
}

void llama_expert_hotstore::log_hit_rate(const std::vector<std::pair<int, ggml_tensor *>> & moe_sel) {
    if (moe_sel.empty() || !is_filled) {
        return;
    }
    size_t hits = 0, total = 0, distinct = 0, distinct_cold = 0;
    int    n_layers_seen = 0;
    int    n_tokens      = 0;
    std::vector<uint8_t> seen(n_experts);
    std::vector<int32_t> ids;
    for (const auto & kv : moe_sel) {
        const int il = kv.first;
        const int64_t n = llama_expert_read_sel_ids(kv.second, ids);
        if (n == 0) {
            continue;
        }
        std::fill(seen.begin(), seen.end(), 0);
        for (size_t i = 0; i < ids.size(); i++) {
            const int32_t id = ids[i];
            if (id >= 0 && id < n_experts) {
                total++;
                seen[id] = 1;
                if (slot_of(il, id) >= 0) {
                    hits++;
                }
            }
        }
        for (int e = 0; e < n_experts; e++) {
            if (seen[e]) {
                distinct++;
                if (slot_of(il, e) < 0) {
                    distinct_cold++;
                }
            }
        }
        n_tokens = (int) n;
        n_layers_seen++;
    }
    if (total > 0) {
        // union is the mean number of distinct experts one layer touches in this batch. it bounds
        // what a store of S slots can save: at most S/union of the expert reads, whatever the
        // ranking. it grows with the batch, so a wide draft shrinks the store's ceiling.
        // cold is the part of that union no slot holds, summed over layers - what the batch
        // actually costs on the host bus, and what the adaptive draft width regresses against.
        LLAMA_LOG("=== expert hot hit rate: %zu/%zu = %.1f%% (m=%d, union %.1f/%d) cold %zu ===\n",
                hits, total, 100.0f * (float) hits / (float) total,
                n_tokens, (float) distinct / (float) n_layers_seen, n_experts, distinct_cold);
    }
}

void llama_expert_hotstore::log() const {
    LLAMA_LOG("=== Expert hotstore sizing (S=%d) ===\n", hot_s);
    size_t total = 0;
    for (int il = 0; il < n_layers; il++) {
        total += bytes_per_slot[il];
        LLAMA_LOG("  layer %3d: bytes/slot = %zu\n", il, bytes_per_slot[il]);
    }
    LLAMA_LOG("  total bytes/slot across all layers = %zu (%zu MiB)\n",
        total, total / (1024 * 1024));
    if (buf) {
        LLAMA_LOG("  GPU hot store allocated: %s, %zu bytes (%zu MiB) for %d slices (%d expert, no pad)\n",
            ggml_backend_buffer_name(buf.get()),
            ggml_backend_buffer_get_size(buf.get()),
            ggml_backend_buffer_get_size(buf.get()) / (1024 * 1024),
            hot_s, hot_s);
    } else if (hot_s > 0) {
        LLAMA_LOG("  hot store DISABLED (%d slots requested)\n", hot_s);
    }
}
