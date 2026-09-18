// Re-pack selected PTQ1_0 tensors of a GGUF as Q4_0, losslessly.
//
// A PTQ1_0 block is 128 trits {-1, 0, +1} with one fp16 scale d. Each group of 32 of them maps to
// one Q4_0 block with the same d and q = trit + 8, so d * (q - 8) gives back the exact value.
// llama-quantize can not do this: its Q4_0 scale is max/-8, which turns +d into 7/8 d.
//
// Use it for the tensors that end up on the CPU: there is no SIMD PTQ1_0 kernel for x86, and Q4_0
// runs at memory bandwidth.
//
// usage: llama-ternary-repack <in.gguf> <out.gguf> <tensor-name-regex>

#include "ggml.h"
#include "gguf.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <regex>
#include <string>
#include <vector>

// Q4_0 block: fp16 d, then 16 bytes, low nibbles hold elements 0..15 and high nibbles 16..31
static constexpr int QK = 32;

static bool repack_q4_0(const ggml_tensor * src, ggml_tensor * dst) {
    const int64_t n = ggml_nelements(src);
    GGML_ASSERT(ggml_blck_size(GGML_TYPE_Q4_0) == QK && n % QK == 0);

    std::vector<float> f(n);
    ggml_get_type_traits(src->type)->to_float(src->data, f.data(), n);

    uint8_t * out = (uint8_t *) dst->data;
    const size_t bs = ggml_type_size(GGML_TYPE_Q4_0);

    for (int64_t ib = 0; ib < n / QK; ++ib) {
        const float * x = f.data() + ib*QK;

        float a = 0.0f;
        for (int j = 0; j < QK; ++j) {
            a = std::max(a, std::fabs(x[j]));
        }

        uint8_t q[QK];
        for (int j = 0; j < QK; ++j) {
            if (x[j] != 0.0f && std::fabs(x[j]) != a) {
                fprintf(stderr, "%s: block %lld is not ternary (%g vs %g)\n", src->name, (long long) ib, x[j], a);
                return false;
            }
            q[j] = (uint8_t) (8 + (x[j] > 0.0f) - (x[j] < 0.0f));
        }

        uint8_t * blk = out + ib*bs;
        const ggml_fp16_t d = ggml_fp32_to_fp16(a);
        memcpy(blk, &d, sizeof(d));
        for (int j = 0; j < QK/2; ++j) {
            blk[sizeof(d) + j] = q[j] | (q[j + QK/2] << 4);
        }
    }

    // round trip must be exact, including the fp16 scale
    std::vector<float> g(n);
    ggml_get_type_traits(GGML_TYPE_Q4_0)->to_float(dst->data, g.data(), n);
    if (memcmp(f.data(), g.data(), n*sizeof(float)) != 0) {
        fprintf(stderr, "%s: round trip is not exact\n", src->name);
        return false;
    }
    return true;
}

int main(int argc, char ** argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s <in.gguf> <out.gguf> <tensor-name-regex>\n", argv[0]);
        return 1;
    }
    const std::regex re(argv[3]);

    ggml_context * ctx_src = nullptr;
    gguf_context * gg_src  = gguf_init_from_file(argv[1], { /*no_alloc =*/ false, /*ctx =*/ &ctx_src });
    if (!gg_src) {
        fprintf(stderr, "failed to read %s\n", argv[1]);
        return 1;
    }

    // size the context for the new tensors
    size_t mem = 0;
    int    n_sel = 0;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx_src); t; t = ggml_get_next_tensor(ctx_src, t)) {
        if (t->type == GGML_TYPE_PTQ1_0 && std::regex_search(t->name, re)) {
            mem += ggml_row_size(GGML_TYPE_Q4_0, ggml_nelements(t)) + ggml_tensor_overhead();
            n_sel++;
        }
    }
    ggml_context * ctx_dst = ggml_init({ mem + ggml_tensor_overhead(), nullptr, false });

    gguf_context * gg_dst = gguf_init_empty();
    gguf_set_kv(gg_dst, gg_src);

    size_t bytes_in = 0, bytes_out = 0;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx_src); t; t = ggml_get_next_tensor(ctx_src, t)) {
        // ctx_src also holds the blob with the raw file data, it is not a model tensor
        if (gguf_find_tensor(gg_src, t->name) < 0) {
            continue;
        }
        if (t->type == GGML_TYPE_PTQ1_0 && std::regex_search(t->name, re)) {
            ggml_tensor * q = ggml_new_tensor(ctx_dst, GGML_TYPE_Q4_0, GGML_MAX_DIMS, t->ne);
            ggml_set_name(q, t->name);
            if (!repack_q4_0(t, q)) {
                return 1;
            }
            bytes_in  += ggml_nbytes(t);
            bytes_out += ggml_nbytes(q);
            gguf_add_tensor(gg_dst, q);
        } else {
            gguf_add_tensor(gg_dst, t);
        }
    }

    fprintf(stderr, "repacked %d tensors, %.1f MiB PTQ1_0 -> %.1f MiB Q4_0, writing %s\n",
            n_sel, bytes_in/1048576.0, bytes_out/1048576.0, argv[2]);

    if (!gguf_write_to_file(gg_dst, argv[2], false)) {
        fprintf(stderr, "failed to write %s\n", argv[2]);
        return 1;
    }

    gguf_free(gg_dst);
    gguf_free(gg_src);
    ggml_free(ctx_dst);
    ggml_free(ctx_src);
    return 0;
}
