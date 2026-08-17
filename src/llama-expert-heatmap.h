#pragma once

#include <vector>
#include <cstdint>
#include <utility>

struct ggml_tensor;

// read a moe_sel_experts tensor into [n_expert_used * n_tokens] host ids.
// ggml_argsort_top_k returns a view of the full [n_expert, n_tokens] argsort, so its
// rows are n_expert apart, not n_expert_used. one read per row, never one flat read.
// returns n_tokens, or 0 if the tensor holds no readable ids.
int64_t llama_expert_read_sel_ids(const ggml_tensor * t, std::vector<int32_t> & ids);

struct llama_expert_heatmap {
    int n_layers;
    int n_experts;
    int hot_s;
    float decay_rate;
    int   log_period;
    int64_t tokens_total; // real tokens seen (not multiplied by layers)

    std::vector<float> heat;

    llama_expert_heatmap(int n_layers, int n_experts,
                         float decay_rate = 0.99f,
                         int log_period = 100,
                         int hot_s = 0);

    void update(int layer_idx, const int32_t * expert_ids, int n_expert_used, int n_tokens);
    void update_from_graph(const std::vector<std::pair<int, ggml_tensor *>> & moe_sel_experts);
    void decay_all();
    void log() const;

    float get_score(int layer_idx, int expert_id) const;
    std::vector<int> get_top_s(int layer_idx, int s) const;
};
