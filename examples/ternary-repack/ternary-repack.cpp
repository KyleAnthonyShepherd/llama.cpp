// Re-pack selected PTQ1_0 tensors of a GGUF as Q4_0 or PQ2_0, losslessly.
//
// A PTQ1_0 block is 128 trits {-1, 0, +1} with one fp16 scale d. Both targets keep d and store
// the trit as a small unsigned code, so d * (code - offset) gives back the exact value:
//   q4_0 : 4 blocks of 32, code = trit + 8. llama-quantize can not do this: its Q4_0 scale is
//          max/-8, which turns +d into 7/8 d.
//   pq2_0: 1 block of 128, code = trit + 1 in 2-bit slots (element e at byte e/4, bits 2*(e%4)).
//
// Use it for the tensors that end up on the CPU: the x86 PTQ1_0 dot is scalar, while Q4_0 and
// PQ2_0 have SIMD kernels.
//
// usage: llama-ternary-repack <in.gguf> <out.gguf> <tensor-name-regex> [q4_0|pq2_0]

#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <regex>
#include <string>
#include <vector>

// trit of one value given the group's max |x|, or -2 if it is not in {-a, 0, a}
static int trit(float x, float a) {
    if (x == 0.0f) {
        return 0;
    }
    if (std::fabs(x) != a) {
        return -2;
    }
    return x > 0.0f ? 1 : -1;
}

static bool repack(const ggml_tensor * src, ggml_tensor * dst) {
    const int64_t n  = ggml_nelements(src);
    const int     qk = (int) ggml_blck_size(dst->type);
    const size_t  bs = ggml_type_size(dst->type);
    GGML_ASSERT(n % qk == 0);
    GGML_ASSERT(dst->type == GGML_TYPE_Q4_0 ? qk == 32 && bs == 18 : qk == 128 && bs == 34);

    std::vector<float> f(n);
    ggml_get_type_traits(src->type)->to_float(src->data, f.data(), n);

    uint8_t * out = (uint8_t *) dst->data;
    for (int64_t ib = 0; ib < n / qk; ++ib) {
        const float * x = f.data() + ib*qk;

        float a = 0.0f;
        for (int j = 0; j < qk; ++j) {
            a = std::max(a, std::fabs(x[j]));
        }

        // both layouts: fp16 d first, then the codes
        uint8_t * blk = out + ib*bs;
        const ggml_fp16_t d = ggml_fp32_to_fp16(a);
        memcpy(blk, &d, sizeof(d));
        uint8_t * qs = blk + sizeof(d);
        memset(qs, 0, bs - sizeof(d));

        for (int j = 0; j < qk; ++j) {
            const int t = trit(x[j], a);
            if (t == -2) {
                fprintf(stderr, "%s: block %lld is not ternary (%g vs %g)\n", src->name, (long long) ib, x[j], a);
                return false;
            }
            if (dst->type == GGML_TYPE_Q4_0) {
                // low nibbles hold elements 0..15, high nibbles 16..31
                qs[j % 16] |= (uint8_t) (t + 8) << (j < 16 ? 0 : 4);
            } else {
                qs[j / 4] |= (uint8_t) (t + 1) << (2*(j % 4));
            }
        }
    }

    // round trip must be exact, including the fp16 scale
    std::vector<float> g(n);
    ggml_get_type_traits(dst->type)->to_float(dst->data, g.data(), n);
    if (memcmp(f.data(), g.data(), n*sizeof(float)) != 0) {
        fprintf(stderr, "%s: round trip is not exact\n", src->name);
        return false;
    }
    return true;
}

int main(int argc, char ** argv) {
    if (argc != 4 && argc != 5) {
        fprintf(stderr, "usage: %s <in.gguf> <out.gguf> <tensor-name-regex> [q4_0|pq2_0]\n", argv[0]);
        return 1;
    }
    const std::regex re(argv[3]);

    ggml_type type = GGML_TYPE_Q4_0;
    if (argc == 5) {
        if (strcmp(argv[4], "pq2_0") == 0) {
            type = GGML_TYPE_PQ2_0;
        } else if (strcmp(argv[4], "q4_0") != 0) {
            fprintf(stderr, "unknown target type %s\n", argv[4]);
            return 1;
        }
    }

    ggml_context * ctx_src = nullptr;
    gguf_context * gg_src  = gguf_init_from_file(argv[1], { /*no_alloc =*/ false, /*ctx =*/ &ctx_src });
    if (!gg_src) {
        fprintf(stderr, "failed to read %s\n", argv[1]);
        return 1;
    }

    // ctx_src also holds the blob with the raw file data, it is not a model tensor
    auto selected = [&](const ggml_tensor * t) {
        return gguf_find_tensor(gg_src, t->name) >= 0 && t->type == GGML_TYPE_PTQ1_0 && std::regex_search(t->name, re);
    };

    // size the context for the new tensors
    size_t mem = ggml_tensor_overhead();
    int    n_sel = 0;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx_src); t; t = ggml_get_next_tensor(ctx_src, t)) {
        if (selected(t)) {
            mem += ggml_row_size(type, ggml_nelements(t)) + ggml_tensor_overhead();
            n_sel++;
        }
    }
    ggml_context * ctx_dst = ggml_init({ mem, nullptr, false });

    gguf_context * gg_dst = gguf_init_empty();
    gguf_set_kv(gg_dst, gg_src);

    size_t bytes_in = 0, bytes_out = 0;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx_src); t; t = ggml_get_next_tensor(ctx_src, t)) {
        if (gguf_find_tensor(gg_src, t->name) < 0) {
            continue;
        }
        if (selected(t)) {
            ggml_tensor * q = ggml_new_tensor(ctx_dst, type, GGML_MAX_DIMS, t->ne);
            ggml_set_name(q, t->name);
            if (!repack(t, q)) {
                return 1;
            }
            bytes_in  += ggml_nbytes(t);
            bytes_out += ggml_nbytes(q);
            gguf_add_tensor(gg_dst, q);
        } else {
            gguf_add_tensor(gg_dst, t);
        }
    }

    fprintf(stderr, "repacked %d tensors, %.1f MiB PTQ1_0 -> %.1f MiB %s, writing %s\n",
            n_sel, bytes_in/1048576.0, bytes_out/1048576.0, ggml_type_name(type), argv[2]);

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
