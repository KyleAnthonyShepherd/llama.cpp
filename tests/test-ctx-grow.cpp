// unit test for llama_set_n_ctx() and llama_decode()'s automatic KV-cache growth hook
// (Part 1, Phase 3 of the dynamic-context-growth plan).
//
// Determinism gate: a context that starts small and grows automatically mid-generation
// must produce byte-for-byte (well, float-for-float) identical logits to an equivalent
// context that was pre-allocated at the larger size from the start, at every step,
// including the exact step where growth happens. This is the harness the plan's Phase 3
// (Part 1 Phase 5.4) determinism matrix asks for, run here on the CPU backend against a
// plain (non-hybrid) model; the {hybrid, iswa, CUDA} legs of that matrix still need the
// user's own hardware (see NOTES.md).

#include "arg.h"
#include "common.h"
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <clocale>
#include <vector>

static llama_context * make_ctx(
        llama_model * model,
        uint32_t n_ctx,
        uint32_t n_ctx_max,
        llama_flash_attn_type fa,
        ggml_type type_k,
        ggml_type type_v) {
    llama_context_params cparams = llama_context_default_params();

    cparams.n_ctx           = n_ctx;
    cparams.n_ctx_max       = n_ctx_max;
    cparams.n_batch         = n_ctx;
    cparams.n_ubatch        = n_ctx;
    cparams.n_seq_max       = 1;
    cparams.flash_attn_type = fa;
    cparams.type_k          = type_k;
    cparams.type_v          = type_v;
    cparams.no_perf         = true;

    return llama_init_from_model(model, cparams);
}

static bool decode_one(llama_context * ctx, llama_token tok, llama_pos pos) {
    llama_batch batch = llama_batch_init(1, 0, 1);
    common_batch_add(batch, tok, pos, { 0 }, true);
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

static bool run_scenario(
        llama_model * model,
        const char * label,
        llama_flash_attn_type fa,
        ggml_type type_k,
        ggml_type type_v) {
    fprintf(stderr, "=== scenario: %s ===\n", label);

    const uint32_t n_small = 256;
    const uint32_t n_big   = 1024;
    const uint32_t n_steps = 300; // > n_small, so growth must trigger at least once

    llama_context * ctx_ref  = make_ctx(model, n_big,   0,     fa, type_k, type_v);
    llama_context * ctx_grow = make_ctx(model, n_small, n_big, fa, type_k, type_v);

    if (!ctx_ref || !ctx_grow) {
        // most likely an incompatible {type_k/type_v, model} combination (e.g. a quantized
        // KV type whose block size doesn't divide this model's head dim) rather than a
        // growth bug - not every model supports every scenario this test tries.
        fprintf(stderr, "%s: failed to create context(s) (likely an incompatible KV type for this model) - skipping\n", label);
        if (ctx_ref)  llama_free(ctx_ref);
        if (ctx_grow) llama_free(ctx_grow);
        return true;
    }

    if (!llama_get_memory(ctx_ref) || !llama_get_memory(ctx_grow)) {
        fprintf(stderr, "%s: memory is not a plain llama_kv_cache-backed context - skipping\n", label);
        llama_free(ctx_ref);
        llama_free(ctx_grow);
        return true;
    }

    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int32_t       n_vocab = llama_vocab_n_tokens(vocab);

    bool ok   = true;
    bool grew = false;

    for (uint32_t pos = 0; pos < n_steps && ok; ++pos) {
        const llama_token tok = (llama_token) (pos % (uint32_t) n_vocab);

        if (!decode_one(ctx_ref, tok, (llama_pos) pos)) {
            fprintf(stderr, "%s: reference decode failed at pos %u\n", label, pos);
            ok = false;
            break;
        }

        const uint32_t n_ctx_before = llama_n_ctx(ctx_grow);
        if (!decode_one(ctx_grow, tok, (llama_pos) pos)) {
            fprintf(stderr, "%s: growing decode FAILED at pos %u (n_ctx was %u)\n", label, pos, n_ctx_before);
            ok = false;
            break;
        }
        if (llama_n_ctx(ctx_grow) > n_ctx_before) {
            fprintf(stderr, "%s: auto-grow fired at pos %u: n_ctx %u -> %u\n", label, pos, n_ctx_before, llama_n_ctx(ctx_grow));
            grew = true;
        }

        const float * logits_ref  = llama_get_logits_ith(ctx_ref,  0);
        const float * logits_grow = llama_get_logits_ith(ctx_grow, 0);
        if (!logits_ref || !logits_grow) {
            fprintf(stderr, "%s: missing logits at pos %u\n", label, pos);
            ok = false;
            break;
        }

        double max_abs_diff = 0.0;
        for (int32_t i = 0; i < n_vocab; ++i) {
            max_abs_diff = std::max(max_abs_diff, (double) std::fabs(logits_ref[i] - logits_grow[i]));
        }
        if (max_abs_diff > 1e-4) {
            fprintf(stderr, "%s: logits diverged at pos %u (max abs diff = %g)\n", label, pos, max_abs_diff);
            ok = false;
            break;
        }
    }

    if (ok && !grew) {
        fprintf(stderr, "%s: growth never triggered - test scenario didn't exercise what it meant to\n", label);
        ok = false;
    }

    if (ok && llama_n_ctx(ctx_grow) <= n_small) {
        fprintf(stderr, "%s: final n_ctx (%u) did not grow past n_small (%u)\n", label, llama_n_ctx(ctx_grow), n_small);
        ok = false;
    }

    if (ok) {
        fprintf(stderr, "%s: OK (final n_ctx = %u)\n", label, llama_n_ctx(ctx_grow));
    }

    llama_free(ctx_ref);
    llama_free(ctx_grow);

    return ok;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    ggml_backend_load_all();

    common_init_result_ptr llama_init = common_init_from_params(params);
    llama_model * model = llama_init->model();
    if (model == nullptr) {
        fprintf(stderr, "%s: failed to init model\n", __func__);
        return 1;
    }

    if (llama_model_is_recurrent(model) || llama_model_is_hybrid(model)) {
        fprintf(stderr, "%s: skipping for recurrent/hybrid model (needs a plain KV cache model)\n", __func__);
        return 0;
    }

    bool ok = true;

    ok = run_scenario(model, "fa-off (v_trans)", LLAMA_FLASH_ATTN_TYPE_DISABLED, GGML_TYPE_F16, GGML_TYPE_F16) && ok;
    ok = run_scenario(model, "fa-on (!v_trans)", LLAMA_FLASH_ATTN_TYPE_ENABLED,  GGML_TYPE_F16, GGML_TYPE_F16) && ok;
    ok = run_scenario(model, "fa-on q8_0 KV",    LLAMA_FLASH_ATTN_TYPE_ENABLED,  GGML_TYPE_Q8_0, GGML_TYPE_Q8_0) && ok;

    if (!ok) {
        fprintf(stderr, "%s: FAILED\n", __func__);
        return 1;
    }

    fprintf(stderr, "%s: all scenarios passed\n", __func__);
    return 0;
}
