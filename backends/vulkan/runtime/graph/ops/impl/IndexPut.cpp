/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/DimUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/TensorUtils.h>

#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

struct IndexPutParams final {
  int32_t gpu_dim;
  int32_t stride;
};

IndexPutParams create_index_put_params(
    ComputeGraph& graph,
    const int64_t dim_idx,
    const ValueRef out) {
  if (dim_idx == kWidth4D) {
    return {0, 1};
  } else if (dim_idx == kHeight4D) {
    return {1, 1};
  } else if (dim_idx == kBatch4D) {
    const std::vector<int64_t> out_sizes = graph.sizes_of(out);
    int64_t n_channels = dim_at(out_sizes, kChannel4D);
    int64_t stride = utils::div_up_4(n_channels);
    return {2, static_cast<int32_t>(stride)};
  } else {
    VK_THROW("Unexpected dim_idx!");
  }
}

void add_index_put_node(
    ComputeGraph& graph,
    ValueRef out,
    const int64_t dim_idx,
    ValueRef idx,
    ValueRef values) {
  
  IndexPutParams params = create_index_put_params(graph, dim_idx, out);

  std::string kernel_name = "index_put";
  kernel_name.reserve(kShaderNameReserve);
  add_dtype_suffix(kernel_name, graph.dtype_of(out));

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      default_pick_global_wg_size,
      default_pick_local_wg_size,
      {{out, vkapi::kWrite}, {{idx, values}, vkapi::kRead}},
      {graph.sizes_ubo(values), graph.create_params_buffer(params), graph.sizes_ubo(out)},
      // Push Constants
      {},
      // Specialization Constants
      {},
      // Resize Args
      {},
      // Resizing Logic
      nullptr));
}

void index_put(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  ValueRef out = args[0];
  ValueRef indices_list = args[1];
  ValueRef values = args[2];
  ValueRef accumulate = args[3];

  // Check accumulate is false
  bool acc = graph.extract_scalar<bool>(accumulate);
  if (acc) {
    VK_THROW("index_put with accumulate=True is not supported yet");
  }

  // Parse indices list
  std::vector<ValueRef> indices = *graph.get_value_list(indices_list);
  
  // Find the single non-null index
  int64_t dim_idx = -1;
  ValueRef idx = kDummyValueRef;
  
  for (size_t i = 0; i < indices.size(); ++i) {
    if (indices[i] != kDummyValueRef && !graph.val_is_none(indices[i])) {
      if (dim_idx != -1) {
        VK_THROW("index_put only supports a single index tensor for now");
      }
      dim_idx = static_cast<int64_t>(i);
      idx = indices[i];
    }
  }

  if (dim_idx == -1) {
    VK_THROW("index_put requires at least one index tensor");
  }

  // Normalize dim_idx
  const int64_t ndim = graph.dim_of(out);
  dim_idx = normalize(dim_idx, ndim);
  // Convert to DimIndex
  int64_t dim_index = dim_idx < 0 ? dim_idx : dim_idx - ndim;

  if (dim_index == kChannel4D) {
     VK_THROW("index_put on channel dimension is not supported yet");
  } else {
    add_index_put_node(graph, out, dim_index, idx, values);
  }
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(aten.index_put.default, index_put);
}

} // namespace vkcompute
