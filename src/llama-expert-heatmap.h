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

// one batch of router selections, read off the graph and kept on the host so the batch can
// be counted later than it ran
struct llama_expert_sel {
    int     n_expert_used = 0;
    int64_t n_tokens      = 0;

    // layer index -> [n_expert_used * n_tokens] ids
    std::vector<std::pair<int, std::vector<int32_t>>> layers;
};

void llama_expert_read_sel(const std::vector<std::pair<int, ggml_tensor *>> & moe_sel, llama_expert_sel & out);

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

    // apply one batch: decay by n_tokens, add its counts, advance tokens_total. n_tokens can be
    // less than the batch that produced sel, to count only the accepted prefix of a draft.
    void update_batch(const llama_expert_sel & sel, int64_t n_tokens);

    void update_from_graph(const std::vector<std::pair<int, ggml_tensor *>> & moe_sel_experts);
    // decay is per token, not per call: a batch adds one count per token, so decaying once
    // per call would tie the half-life to the batch shape instead of to usage
    void decay_all(int64_t n_tokens);
    void log() const;

    float get_score(int layer_idx, int expert_id) const;
    std::vector<int> get_top_s(int layer_idx, int s) const;
};
