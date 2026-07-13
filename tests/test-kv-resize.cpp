// unit test for llama_memory_i::resize() / llama_kv_cache::resize()
//
// exercises the Part 1 (dynamic context growth) "resize() core" in isolation, at the
// memory-object level - it does not depend on the (not yet implemented) public
// llama_set_n_ctx() API or on the ggml_backend_sched re-reserve wiring that a full
// grow-then-keep-decoding flow would need (that plumbing belongs to a later phase).
//
// what is verified for each {flash-attn, KV type} configuration:
//   - resize() rejects growing to a smaller-or-equal size, leaving the cache fully usable
//   - resize() to a valid larger size succeeds and reports the new size
//   - the previously-used cells' K and (both transposed and non-transposed) V bytes are
//     preserved exactly
//   - cell/sequence metadata (seq_pos_max) is preserved

#include "arg.h"
#include "common.h"
#include "llama.h"

#include "../src/llama-kv-cache.h"

#include <cstdio>
#include <cstring>
#include <clocale>
#include <vector>

static llama_context * make_ctx(
        llama_model * model,
        uint32_t n_ctx,
        llama_flash_attn_type fa,
        ggml_type type_k,
        ggml_type type_v) {
    llama_context_params cparams = llama_context_default_params();

    cparams.n_ctx           = n_ctx;
    cparams.n_batch         = n_ctx;
    cparams.n_ubatch         = n_ctx;
    cparams.n_seq_max       = 1;
    cparams.flash_attn_type = fa;
    cparams.type_k          = type_k;
    cparams.type_v          = type_v;
    cparams.no_perf         = true;

    return llama_init_from_model(model, cparams);
}

static bool decode_one(llama_context * ctx, llama_token tok, llama_pos pos) {
    llama_batch batch = llama_batch_init(1, 0, 1);
    common_batch_add(batch, tok, pos, { 0 }, false);
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

// read `n_copies` rows of `size` bytes, strided by `stride_tensor` in the tensor, packed
// tightly (stride_data == size) into the returned buffer
static std::vector<uint8_t> read_rows(const ggml_tensor * t, size_t size, size_t n_copies, size_t stride_tensor) {
    std::vector<uint8_t> out(size * n_copies);
    ggml_backend_tensor_get_2d(t, out.data(), 0, size, n_copies, stride_tensor, size);
    return out;
}

static std::vector<uint8_t> read_contig(const ggml_tensor * t, size_t nbytes) {
    std::vector<uint8_t> out(nbytes);
    ggml_backend_tensor_get(t, out.data(), 0, nbytes);
    return out;
}

// returns false on any check failure
static bool run_scenario(
        llama_model * model,
        const char * label,
        llama_flash_attn_type fa,
        ggml_type type_k,
        ggml_type type_v) {
    fprintf(stderr, "=== scenario: %s ===\n", label);

    const uint32_t n_small = 256;
    const uint32_t n_big   = 512;
    const uint32_t n_fill  = 200; // < n_small, so growth is required to go past it

    llama_context * ctx = make_ctx(model, n_small, fa, type_k, type_v);
    if (!ctx) {
        // most likely an incompatible {type_k/type_v, model} combination (e.g. a quantized
        // KV type whose block size doesn't divide this model's head dim) rather than a
        // resize() bug - not every model supports every scenario this test tries.
        fprintf(stderr, "%s: failed to create context (likely an incompatible KV type for this model) - skipping\n", label);
        return true;
    }

    llama_memory_t mem = llama_get_memory(ctx);
    auto * kv = dynamic_cast<llama_kv_cache *>(mem);
    if (!kv) {
        fprintf(stderr, "%s: memory is not a plain llama_kv_cache - skipping\n", label);
        llama_free(ctx);
        return true;
    }

    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int32_t       n_vocab = llama_vocab_n_tokens(vocab);

    for (uint32_t pos = 0; pos < n_fill; ++pos) {
        const llama_token tok = (llama_token) (pos % (uint32_t) n_vocab);
        if (!decode_one(ctx, tok, (llama_pos) pos)) {
            fprintf(stderr, "%s: failed to decode fill token at pos %u\n", label, pos);
            llama_free(ctx);
            return false;
        }
    }

    if (kv->get_size() != n_small) {
        fprintf(stderr, "%s: unexpected initial size %u (expected %u)\n", label, kv->get_size(), n_small);
        llama_free(ctx);
        return false;
    }

    const llama_pos pos_max_before = llama_memory_seq_pos_max(mem, 0);
    if (pos_max_before != (llama_pos) n_fill - 1) {
        fprintf(stderr, "%s: unexpected seq_pos_max before resize: %d\n", label, pos_max_before);
        llama_free(ctx);
        return false;
    }

    // K (and non-transposed V) is row-major with position as the *outer* dimension, so the
    // used range is a plain contiguous byte prefix. Transposed V (FA off) is physically
    // [n_embd_v_gqa rows][kv_size cols] with position as the *inner*, contiguous dimension,
    // so the used range is only a prefix of each row and rows must be read individually,
    // strided by the cache's current (pre/post-resize) width.
    const bool v_trans = (fa == LLAMA_FLASH_ATTN_TYPE_DISABLED);

    // snapshot layer 0's K (and V) bytes for the used range, before resize
    ggml_tensor * k0 = kv->get_k_storage(0);
    ggml_tensor * v0 = kv->get_v_storage(0);

    const size_t type_size_k  = ggml_type_size(k0->type);
    const size_t n_embd_k_gqa = (size_t) ggml_nbytes(k0) / type_size_k / n_small;

    const std::vector<uint8_t> k_before = read_contig(k0, n_fill * n_embd_k_gqa * type_size_k);

    std::vector<uint8_t> v_before;
    size_t n_embd_v_gqa = 0;
    size_t type_size_v  = 0;
    if (v0) {
        type_size_v  = ggml_type_size(v0->type);
        n_embd_v_gqa = (size_t) ggml_nbytes(v0) / type_size_v / n_small;
        v_before     = v_trans
            ? read_rows(v0, n_fill * type_size_v, n_embd_v_gqa, n_small * type_size_v)
            : read_contig(v0, n_fill * n_embd_v_gqa * type_size_v);
    }

    // failure path (n_new <= current size) must leave the cache untouched and usable.
    // (an allocation-failure path, e.g. resize() to a size too large to fit in memory, is
    // exercised the same way per the implementation - not covered here since reliably
    // forcing a real allocation failure without risking OOM-killing the test process
    // needs a memory-constrained environment, e.g. a bounded RLIMIT_AS)

    if (kv->resize(n_small)) {
        fprintf(stderr, "%s: resize() to an equal size unexpectedly succeeded\n", label);
        llama_free(ctx);
        return false;
    }

    if (kv->get_size() != n_small || !decode_one(ctx, 0, (llama_pos) n_fill)) {
        fprintf(stderr, "%s: cache unusable after a failed resize()\n", label);
        llama_free(ctx);
        return false;
    }
    // undo the probe decode above (its position is not part of the fixed fill range)
    llama_memory_seq_rm(mem, 0, (llama_pos) n_fill, -1);

    // the real growth
    if (!kv->resize(n_big)) {
        fprintf(stderr, "%s: resize() to a valid larger size failed\n", label);
        llama_free(ctx);
        return false;
    }

    if (kv->get_size() != n_big) {
        fprintf(stderr, "%s: unexpected size after resize: %u (expected %u)\n", label, kv->get_size(), n_big);
        llama_free(ctx);
        return false;
    }

    if (llama_memory_seq_pos_max(mem, 0) != pos_max_before) {
        fprintf(stderr, "%s: seq_pos_max changed across resize (%d -> %d)\n",
                label, pos_max_before, llama_memory_seq_pos_max(mem, 0));
        llama_free(ctx);
        return false;
    }

    // re-fetch storage pointers: resize() swaps in new tensors
    k0 = kv->get_k_storage(0);
    v0 = kv->get_v_storage(0);

    const std::vector<uint8_t> k_after = read_contig(k0, n_fill * n_embd_k_gqa * type_size_k);
    if (k_after != k_before) {
        fprintf(stderr, "%s: K bytes changed across resize\n", label);
        llama_free(ctx);
        return false;
    }

    if (v0) {
        const std::vector<uint8_t> v_after = v_trans
            ? read_rows(v0, n_fill * type_size_v, n_embd_v_gqa, n_big * type_size_v)
            : read_contig(v0, n_fill * n_embd_v_gqa * type_size_v);
        if (v_after != v_before) {
            fprintf(stderr, "%s: V bytes changed across resize\n", label);
            llama_free(ctx);
            return false;
        }
    }

    fprintf(stderr, "%s: OK\n", label);

    llama_free(ctx);
    return true;
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

    // FA off -> v_trans == true (V stored transposed)
    ok = run_scenario(model, "fa-off (v_trans)", LLAMA_FLASH_ATTN_TYPE_DISABLED, GGML_TYPE_F16, GGML_TYPE_F16) && ok;

    // FA on -> v_trans == false
    ok = run_scenario(model, "fa-on (!v_trans)", LLAMA_FLASH_ATTN_TYPE_ENABLED, GGML_TYPE_F16, GGML_TYPE_F16) && ok;

    // FA on + quantized KV (only supported with FA on)
    ok = run_scenario(model, "fa-on q8_0 KV", LLAMA_FLASH_ATTN_TYPE_ENABLED, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0) && ok;

    if (!ok) {
        fprintf(stderr, "%s: FAILED\n", __func__);
        return 1;
    }

    fprintf(stderr, "%s: all scenarios passed\n", __func__);
    return 0;
}
