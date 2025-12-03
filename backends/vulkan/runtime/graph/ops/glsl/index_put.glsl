/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#define PRECISION ${PRECISION}

#define VEC4_T ${texel_load_type(DTYPE, STORAGE)}

${define_required_extensions(DTYPE)}
${define_active_storage_type(STORAGE)}

layout(std430) buffer;

#include "indexing_utils.h"

${layout_declare_tensor(0, "w", "t_out", DTYPE, STORAGE)}
${layout_declare_tensor(1, "r", "t_idx", "int", STORAGE)}
${layout_declare_tensor(2, "r", "t_values", DTYPE, STORAGE)}
${layout_declare_ubo(3, "ivec4", "sizes")}
${layout_declare_ubo(4, "int", "gpu_dim", "int", "stride")}
${layout_declare_ubo(5, "ivec4", "out_sizes")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(constant_id = 3) const int packed_dim = C_DIM;

void main() {
  const ivec3 val_pos = ivec3(gl_GlobalInvocationID);

  if (pos_out_of_bounds(val_pos, sizes, packed_dim)) {
    return;
  }

#ifdef USING_BUFFER
  // Buffer logic: iterate over 4 elements of the texel
  ivec4 val_nchw_base = to_tensor_idx(val_pos, sizes, packed_dim);

  for (int i = 0; i < 4; ++i) {
    ivec4 elem_nchw = val_nchw_base;
    elem_nchw[packed_dim] += i;

    if (elem_nchw[packed_dim] >= sizes[packed_dim]) {
      continue;
    }

    int elem_val_bufi = tidx_to_nchwi(elem_nchw, sizes);
    
    // Get index from t_idx
    int idx_idx = elem_nchw[gpu_dim];
    int out_dim_idx = t_idx[idx_idx];

    // Calculate out position
    ivec4 out_nchw = elem_nchw;
    out_nchw[gpu_dim] = out_dim_idx;

    int out_bufi = tidx_to_nchwi(out_nchw, out_sizes);

    t_out[out_bufi] = t_values[elem_val_bufi];
  }

#else
  // Texture logic
  const int idx_index = val_pos[gpu_dim] / stride;
  const int within_stride = val_pos[gpu_dim] % stride;

  int out_dim_idx = texelFetch(t_idx, ivec3(idx_index, 0, 0), 0).x;

  ivec3 out_pos = val_pos;
  out_pos[gpu_dim] = out_dim_idx * stride + within_stride;

  VEC4_T val = texelFetch(t_values, val_pos, 0);

  write_texel(t_out, out_pos, ${"uvec4" if DTYPE == "bool" else ""}(val));
#endif
}
