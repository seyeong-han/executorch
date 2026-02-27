# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.24)
project(qwen3_5_runner)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)

include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)

set(_common_include_directories ${EXECUTORCH_ROOT}/..)

set(gflags_DIR ${CMAKE_CURRENT_BINARY_DIR}/../../../third-party/gflags)
find_package(gflags REQUIRED)

list(APPEND CMAKE_FIND_ROOT_PATH ${CMAKE_CURRENT_BINARY_DIR}/../../..)
find_package(executorch CONFIG REQUIRED FIND_ROOT_PATH_BOTH)
get_target_property(_executorch_imported executorch IMPORTED)
if(NOT _executorch_imported)
  executorch_target_link_options_shared_lib(executorch)
endif()

set(link_libraries executorch gflags)

if(TARGET optimized_native_cpu_ops_lib)
  list(APPEND link_libraries optimized_native_cpu_ops_lib cpublas eigen_blas)
  get_target_property(_is_imported optimized_native_cpu_ops_lib IMPORTED)
  if(NOT _is_imported)
    executorch_target_link_options_shared_lib(optimized_native_cpu_ops_lib)
  endif()
endif()

if(NOT EXECUTORCH_BUILD_CUDA)
  list(APPEND link_libraries quantized_ops_lib custom_ops)
  executorch_target_link_options_shared_lib(quantized_ops_lib)
  executorch_target_link_options_shared_lib(custom_ops)
endif()

# XNNPACK
if(TARGET xnnpack_backend)
  set(xnnpack_backend_libs xnnpack_backend XNNPACK xnnpack-microkernels-prod)
  if(TARGET kleidiai)
    list(APPEND xnnpack_backend_libs kleidiai)
  endif()
  list(APPEND link_libraries ${xnnpack_backend_libs})
  get_target_property(_xnnpack_imported xnnpack_backend IMPORTED)
  if(NOT _xnnpack_imported)
    executorch_target_link_options_shared_lib(xnnpack_backend)
  endif()
endif()

if(ANDROID)
  list(APPEND link_libraries log)
endif()

list(
  APPEND
  link_libraries
  extension_llm_runner
  extension_module
  extension_data_loader
  extension_tensor
  extension_flat_tensor
)

# Metal
if(EXECUTORCH_BUILD_METAL)
  list(APPEND link_libraries metal_backend)
  executorch_target_link_options_shared_lib(metal_backend)
endif()
