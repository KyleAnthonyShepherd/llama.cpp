#pragma once

// Shared helpers between ggml-cpu.c (stock GGML_OP_MUL_MAT_ID) and the
// dedicated MUL_MAT_ID_COLD kernel in ggml-cpu-mul-mat-id-cold.cpp.

#include "ggml-cpu-impl.h"

#include <stdint.h>
#include <stddef.h>

// the rows of each src0 matrix are stored back to back, starting at matrix_row_offs[row_id]
#define MMID_MATRIX_ROW(row_id, i1) matrix_rows[matrix_row_offs[row_id] + (i1)]

struct mmid_row_mapping {
    int32_t i1;
    int32_t i2;
};

#ifdef __cplusplus
extern "C" {
#endif

void * incr_ptr_aligned(void ** p, size_t size, size_t align);

void ggml_compute_forward_mul_mat_id_one_chunk(
    struct ggml_tensor * dst,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const int64_t ir0_start,
    const int64_t ir0_end,
    const int64_t ir1_start,
    const int64_t ir1_end,
    const char * src0_cur,
    const struct mmid_row_mapping * matrix_rows, // rows of the current src0 matrix
    const size_t row_size,
    const bool src1_cont,
    const void * wdata);

void ggml_compute_forward_mul_mat_id_cold(
    const struct ggml_compute_params * params,
          struct ggml_tensor * dst);

#ifdef __cplusplus
}
#endif
