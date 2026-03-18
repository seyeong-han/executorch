/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <Foundation/Foundation.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/backends/apple/metal/runtime/shims/shim_mps.h>
#include <executorch/backends/apple/metal/runtime/shims/et_metal.h>
#include <functional>
#include <unordered_map>

namespace executorch {
namespace backends {
namespace metal {

// Declare the global mapping from et_metal.mm
extern std::unordered_map<void*, id<MTLBuffer>> ptr_to_mtl_buffer;

namespace {

AOTITorchError check_kernel_status(
    const char* api_name,
    ETMetalKernelFunction* function) {
    if (function->hasError()) {
        ET_LOG(
            Error,
            "%s: kernel execution state invalid: %s",
            api_name,
            function->lastError().c_str());
        return Error::Internal;
    }
    return Error::Ok;
}

} // namespace

extern "C" {

// MetalShaderLibrary functions
AOTITorchError aoti_torch_mps_create_shader_library(
    const char* metal_shader_source,
    AOTIMetalShaderLibraryHandle* library_handle) {

    if (!metal_shader_source || !library_handle) {
        ET_LOG(Error, "aoti_torch_mps_create_shader_library: null arguments");
        return Error::InvalidArgument;
    }
    ET_LOG(Info, "aoti_torch_mps_create_shader_library: begin");
    *library_handle = nullptr;

    @autoreleasepool {
        try {
            auto library = std::make_unique<ETMetalShaderLibrary>(std::string(metal_shader_source));
            if (!library->isCompiled()) {
                ET_LOG(
                    Error,
                    "aoti_torch_mps_create_shader_library: shader compilation failed: %s",
                    library->lastError().c_str());
                return Error::Internal;
            }
            auto* raw_library = library.get();

            // Store the unique_ptr to keep the object alive
            storeLibraryHandle(raw_library, std::move(library));

            // Return raw pointer to match existing API
            *library_handle = reinterpret_cast<AOTIMetalShaderLibraryHandle>(raw_library);

            ET_LOG(Debug, "aoti_torch_mps_create_shader_library: Created shader library %p", raw_library);
            return Error::Ok;

        } catch (const std::exception& e) {
            ET_LOG(Error, "aoti_torch_mps_create_shader_library exception: %s", e.what());
            return Error::Internal;
        } catch (...) {
            ET_LOG(Error, "aoti_torch_mps_create_shader_library: unknown exception");
            return Error::Internal;
        }
    }
}

AOTITorchError aoti_torch_mps_delete_shader_library(
    AOTIMetalShaderLibraryHandle library_handle) {

    if (!library_handle) {
        ET_LOG(Error, "aoti_torch_mps_delete_shader_library: null library handle");
        return Error::InvalidArgument;
    }

    try {
        auto* library = reinterpret_cast<ETMetalShaderLibrary*>(library_handle);
        if (removeLibraryHandle(library)) {
            ET_LOG(Debug, "aoti_torch_mps_delete_shader_library: Deleted shader library %p", library);
        } else {
            ET_LOG(Error, "aoti_torch_mps_delete_shader_library: Library not found in storage");
            return Error::InvalidArgument;
        }

        return Error::Ok;

    } catch (const std::exception& e) {
        ET_LOG(Error, "aoti_torch_mps_delete_shader_library exception: %s", e.what());
        return Error::Internal;
    } catch (...) {
        ET_LOG(Error, "aoti_torch_mps_delete_shader_library: unknown exception");
        return Error::Internal;
    }
}

AOTITorchError aoti_torch_mps_get_kernel_function(
    AOTIMetalShaderLibraryHandle library_handle,
    const char* kernel_name,
    AOTIMetalKernelFunctionHandle* function_handle) {

    if (!library_handle || !kernel_name || !function_handle) {
        ET_LOG(Error, "aoti_torch_mps_get_kernel_function: null arguments");
        return Error::InvalidArgument;
    }
    ET_LOG(Info, "aoti_torch_mps_get_kernel_function: kernel=%s", kernel_name);
    *function_handle = nullptr;

    @autoreleasepool {
        try {
            auto* library = reinterpret_cast<ETMetalShaderLibrary*>(library_handle);
            auto function_shared_ptr = library->getKernelFunction(std::string(kernel_name));
            if (!function_shared_ptr) {
                ET_LOG(
                    Error,
                    "aoti_torch_mps_get_kernel_function: Failed to get kernel function '%s': %s",
                    kernel_name,
                    library->lastError().c_str());
                return Error::Internal;
            }

            auto* raw_function = function_shared_ptr.get();

            // Store the shared_ptr to keep the object alive
            storeFunctionHandle(raw_function, function_shared_ptr);

            // Return raw pointer to match existing API
            *function_handle = reinterpret_cast<AOTIMetalKernelFunctionHandle>(raw_function);

            ET_LOG(Debug, "aoti_torch_mps_get_kernel_function: Got kernel function '%s' -> %p", kernel_name, raw_function);
            return Error::Ok;

        } catch (const std::exception& e) {
            ET_LOG(Error, "aoti_torch_mps_get_kernel_function exception: %s", e.what());
            return Error::Internal;
        } catch (...) {
            ET_LOG(Error, "aoti_torch_mps_get_kernel_function: unknown exception");
            return Error::Internal;
        }
    }
}

AOTITorchError aoti_torch_mps_start_encoding(
    AOTIMetalKernelFunctionHandle func) {

    if (!func) {
        ET_LOG(Error, "aoti_torch_mps_start_encoding: null function handle");
        return Error::InvalidArgument;
    }
    ET_LOG(Info, "aoti_torch_mps_start_encoding");

    @autoreleasepool {
        try {
            auto* function = reinterpret_cast<ETMetalKernelFunction*>(func);
            function->clearError();
            function->startEncoding();
            auto kernel_status =
                check_kernel_status("aoti_torch_mps_start_encoding", function);
            if (kernel_status != Error::Ok) {
                return kernel_status;
            }

            ET_LOG(Debug, "aoti_torch_mps_start_encoding: Started encoding for function %p", function);
            return Error::Ok;

        } catch (const std::exception& e) {
            ET_LOG(Error, "aoti_torch_mps_start_encoding exception: %s", e.what());
            return Error::Internal;
        } catch (...) {
            ET_LOG(Error, "aoti_torch_mps_start_encoding: unknown exception");
            return Error::Internal;
        }
    }
}

AOTITorchError aoti_torch_mps_set_arg_tensor(
    AOTIMetalKernelFunctionHandle func,
    unsigned idx,
    AOTITensorHandle tensor) {

    if (!func || !tensor) {
        ET_LOG(Error, "aoti_torch_mps_set_arg_tensor: null function handle or tensor");
        return Error::InvalidArgument;
    }
    ET_LOG(Info, "aoti_torch_mps_set_arg_tensor: idx=%u", idx);

    @autoreleasepool {
        try {
            auto* function = reinterpret_cast<ETMetalKernelFunction*>(func);
            auto* et_tensor = reinterpret_cast<executorch::runtime::etensor::Tensor*>(tensor);

            function->setArg(idx, *et_tensor);
            auto kernel_status =
                check_kernel_status("aoti_torch_mps_set_arg_tensor", function);
            if (kernel_status != Error::Ok) {
                return kernel_status;
            }

            ET_LOG(Debug, "aoti_torch_mps_set_arg_tensor: Set tensor argument at index %u", idx);
            return Error::Ok;

        } catch (const std::exception& e) {
            ET_LOG(Error, "aoti_torch_mps_set_arg_tensor exception: %s", e.what());
            return Error::Internal;
        } catch (...) {
            ET_LOG(Error, "aoti_torch_mps_set_arg_tensor: unknown exception");
            return Error::Internal;
        }
    }
}

AOTITorchError aoti_torch_mps_set_arg_int(
    AOTIMetalKernelFunctionHandle func,
    unsigned idx,
    int64_t val) {

    if (!func) {
        ET_LOG(Error, "aoti_torch_mps_set_arg_int: null function handle");
        return Error::InvalidArgument;
    }
    ET_LOG(Info, "aoti_torch_mps_set_arg_int: idx=%u val=%lld", idx, val);

    try {
        auto* function = reinterpret_cast<ETMetalKernelFunction*>(func);
        function->setArg(idx, val);
        auto kernel_status =
            check_kernel_status("aoti_torch_mps_set_arg_int", function);
        if (kernel_status != Error::Ok) {
            return kernel_status;
        }

        ET_LOG(Debug, "aoti_torch_mps_set_arg_int: Set int64_t value %lld at index %u", val, idx);
        return Error::Ok;

    } catch (const std::exception& e) {
        ET_LOG(Error, "aoti_torch_mps_set_arg_int exception: %s", e.what());
        return Error::Internal;
    } catch (...) {
        ET_LOG(Error, "aoti_torch_mps_set_arg_int: unknown exception");
        return Error::Internal;
    }
}

// Pure C dispatch functions - single value versions
AOTITorchError aoti_torch_mps_dispatch_single(
    AOTIMetalKernelFunctionHandle func,
    uint64_t length) {

    if (!func) {
        ET_LOG(Error, "aoti_torch_mps_dispatch_single: null function handle");
        return Error::InvalidArgument;
    }
    ET_LOG(Info, "aoti_torch_mps_dispatch_single: length=%llu", (unsigned long long)length);

    try {
        auto* function = reinterpret_cast<ETMetalKernelFunction*>(func);
        function->dispatchSingle(length);
        auto kernel_status =
            check_kernel_status("aoti_torch_mps_dispatch_single", function);
        if (kernel_status != Error::Ok) {
            return kernel_status;
        }

        ET_LOG(Debug, "aoti_torch_mps_dispatch_single: Dispatched function %p with length %llu", function, length);
        return Error::Ok;

    } catch (const std::exception& e) {
        ET_LOG(Error, "aoti_torch_mps_dispatch_single exception: %s", e.what());
        return Error::Internal;
    } catch (...) {
        ET_LOG(Error, "aoti_torch_mps_dispatch_single: unknown exception");
        return Error::Internal;
    }
}

AOTITorchError aoti_torch_mps_dispatch_single_with_group_size(
    AOTIMetalKernelFunctionHandle func,
    uint64_t length,
    uint64_t group_size) {

    if (!func) {
        ET_LOG(Error, "aoti_torch_mps_dispatch_single_with_group_size: null function handle");
        return Error::InvalidArgument;
    }
    ET_LOG(
        Info,
        "aoti_torch_mps_dispatch_single_with_group_size: length=%llu group_size=%llu",
        (unsigned long long)length,
        (unsigned long long)group_size);

    try {
        auto* function = reinterpret_cast<ETMetalKernelFunction*>(func);
        function->dispatchSingleWithGroupSize(length, group_size);
        auto kernel_status = check_kernel_status(
            "aoti_torch_mps_dispatch_single_with_group_size", function);
        if (kernel_status != Error::Ok) {
            return kernel_status;
        }

        ET_LOG(Debug, "aoti_torch_mps_dispatch_single_with_group_size: Dispatched function %p with length %llu, group size %llu", function, length, group_size);
        return Error::Ok;

    } catch (const std::exception& e) {
        ET_LOG(Error, "aoti_torch_mps_dispatch_single_with_group_size exception: %s", e.what());
        return Error::Internal;
    } catch (...) {
        ET_LOG(Error, "aoti_torch_mps_dispatch_single_with_group_size: unknown exception");
        return Error::Internal;
    }
}

// Pure C dispatch functions - array versions
AOTITorchError aoti_torch_mps_dispatch_array(
    AOTIMetalKernelFunctionHandle func,
    const uint64_t* length,
    size_t length_size) {

    if (!func) {
        ET_LOG(Error, "aoti_torch_mps_dispatch_array: null function handle");
        return Error::InvalidArgument;
    }
    ET_LOG(Info, "aoti_torch_mps_dispatch_array: rank=%zu", length_size);

    if (!length) {
        ET_LOG(Error, "aoti_torch_mps_dispatch_array_with_group_size: null length pointer");
        return Error::InvalidArgument;
    }

    try {
        auto* function = reinterpret_cast<ETMetalKernelFunction*>(func);
        function->dispatchArray(length, length_size);
        auto kernel_status =
            check_kernel_status("aoti_torch_mps_dispatch_array", function);
        if (kernel_status != Error::Ok) {
            return kernel_status;
        }

        ET_LOG(Debug, "aoti_torch_mps_dispatch_array: Dispatched function %p with %zu dimensions", function, length_size);
        return Error::Ok;

    } catch (const std::exception& e) {
        ET_LOG(Error, "aoti_torch_mps_dispatch_array exception: %s", e.what());
        return Error::Internal;
    } catch (...) {
        ET_LOG(Error, "aoti_torch_mps_dispatch_array: unknown exception");
        return Error::Internal;
    }
}

AOTITorchError aoti_torch_mps_dispatch_array_with_group_size(
    AOTIMetalKernelFunctionHandle func,
    const uint64_t* length,
    size_t length_size,
    const uint64_t* group_size,
    size_t group_size_size) {

    if (!func) {
        ET_LOG(Error, "aoti_torch_mps_dispatch_array_with_group_size: null function handle");
        return Error::InvalidArgument;
    }
    ET_LOG(Info, "aoti_torch_mps_dispatch_array_with_group_size: rank=%zu", length_size);

    if (!length) {
        ET_LOG(Error, "aoti_torch_mps_dispatch_array_with_group_size: null length pointer");
        return Error::InvalidArgument;
    }

    try {
        auto* function = reinterpret_cast<ETMetalKernelFunction*>(func);
        function->dispatchArrayWithGroupSize(length, length_size, group_size, group_size_size);
        auto kernel_status = check_kernel_status(
            "aoti_torch_mps_dispatch_array_with_group_size", function);
        if (kernel_status != Error::Ok) {
            return kernel_status;
        }

        ET_LOG(Debug, "aoti_torch_mps_dispatch_array_with_group_size: Dispatched function %p with %zu dimensions", function, length_size);
        return Error::Ok;

    } catch (const std::exception& e) {
        ET_LOG(Error, "aoti_torch_mps_dispatch_array_with_group_size exception: %s", e.what());
        return Error::Internal;
    } catch (...) {
        ET_LOG(Error, "aoti_torch_mps_dispatch_array_with_group_size: unknown exception");
        return Error::Internal;
    }
}

AOTITorchError aoti_torch_mps_malloc(void** buffer, size_t num_bytes) {
    if (num_bytes == 0) {
        *buffer = nullptr;
        return Error::Ok;
    }
    ET_LOG(Info, "aoti_torch_mps_malloc: bytes=%zu", num_bytes);

    if (!buffer) {
        ET_LOG(Error, "aoti_torch_mps_malloc: null buffer pointer");
        return Error::InvalidArgument;
    }

    @autoreleasepool {
        try {
            id<MTLDevice> device = get_metal_device();
            if (!device) {
                ET_LOG(Error, "aoti_torch_mps_malloc: Failed to get Metal device");
                return Error::Internal;
            }

            id<MTLBuffer> metal_buffer = [device newBufferWithLength:num_bytes
                                                             options:MTLResourceCPUCacheModeWriteCombined | MTLResourceStorageModeShared];
            if (!metal_buffer) {
                ET_LOG(Error, "aoti_torch_mps_malloc: Failed to allocate Metal buffer of size %zu", num_bytes);
                return Error::Internal;
            }

            // FIX: Return contents pointer, not buffer object
            void* contents_ptr = [metal_buffer contents];
            ptr_to_mtl_buffer[contents_ptr] = metal_buffer;  // Map contents to buffer
            *buffer = contents_ptr;  // Return contents pointer

            ET_LOG(Debug, "aoti_torch_mps_malloc: Allocated Metal buffer %p with contents %p of size %zu",
                   metal_buffer, contents_ptr, num_bytes);
            return Error::Ok;

        } catch (const std::exception& e) {
            ET_LOG(Error, "aoti_torch_mps_malloc exception: %s", e.what());
            return Error::Internal;
        } catch (...) {
            ET_LOG(Error, "aoti_torch_mps_malloc: unknown exception");
            return Error::Internal;
        }
    }
}

AOTITorchError aoti_torch_mps_free(void* ptr) {
    if (!ptr) {
        return Error::Ok;  // Nothing to free
    }

    @autoreleasepool {
        try {
            // FIX: ptr is now the contents pointer, not the buffer object
            // Look up the buffer from the mapping and clean up
            auto it = ptr_to_mtl_buffer.find(ptr);
            if (it != ptr_to_mtl_buffer.end()) {
                id<MTLBuffer> metal_buffer = it->second;
                [metal_buffer release];
                ptr_to_mtl_buffer.erase(it);
                ET_LOG(Debug, "aoti_torch_mps_free: Freed Metal buffer for contents %p", ptr);
            } else {
                ET_LOG(Error, "aoti_torch_mps_free: Buffer not found for contents pointer %p", ptr);
                return Error::InvalidArgument;
            }

            return Error::Ok;

        } catch (const std::exception& e) {
            ET_LOG(Error, "aoti_torch_mps_free exception: %s", e.what());
            return Error::Internal;
        } catch (...) {
            ET_LOG(Error, "aoti_torch_mps_free: unknown exception");
            return Error::Internal;
        }
    }
}

AOTITorchError aoti_torch_mps_memcpy(
    void* buffer,
    size_t constant_offset,
    size_t bytes_read,
    size_t data_size,
    uint8_t* constants_start) {

    if (!buffer || !constants_start) {
        ET_LOG(Error, "aoti_torch_mps_memcpy: null buffer or constants_start");
        return Error::InvalidArgument;
    }
    ET_LOG(
        Info,
        "aoti_torch_mps_memcpy: constant_offset=%zu bytes_read=%zu size=%zu",
        constant_offset,
        bytes_read,
        data_size);

    @autoreleasepool {
        try {
            // FIX: buffer is now the contents pointer, not the buffer object
            auto buffer_pointer = static_cast<uint8_t*>(buffer);

            memcpy(buffer_pointer + constant_offset, constants_start + bytes_read, data_size);

            id<MTLDevice> device = get_metal_device();
            if (!device) {
                ET_LOG(Error, "aoti_torch_mps_memcpy: Failed to get Metal device");
                return Error::Internal;
            }
            id<MTLBuffer> subBuffer = [device newBufferWithBytesNoCopy:buffer_pointer + constant_offset
                                                                length:data_size
                                                               options:MTLResourceCPUCacheModeWriteCombined | MTLResourceStorageModeShared
                                                           deallocator:nil];

            if (constant_offset != 0) {
                ptr_to_mtl_buffer[buffer_pointer + constant_offset] = subBuffer;  // Map contents to buffer
            }

            ET_LOG(Debug, "aoti_torch_mps_memcpy: Copied %zu bytes from offset %zu to buffer offset %zu",
                   data_size, bytes_read, constant_offset);
            return Error::Ok;

        } catch (const std::exception& e) {
            ET_LOG(Error, "aoti_torch_mps_memcpy exception: %s", e.what());
            return Error::Internal;
        } catch (...) {
            ET_LOG(Error, "aoti_torch_mps_memcpy: unknown exception");
            return Error::Internal;
        }
    }
}

AOTITorchError aoti_torch_mps_copy_buffer(
    void* src_buffer,
    void* dst_buffer,
    size_t data_size,
    size_t src_offset,
    size_t dst_offset) {

    if (!src_buffer || !dst_buffer) {
        ET_LOG(Error, "aoti_torch_mps_copy_buffer: null buffer");
        return Error::InvalidArgument;
    }
    ET_LOG(
        Info,
        "aoti_torch_mps_copy_buffer: size=%zu src_offset=%zu dst_offset=%zu",
        data_size,
        src_offset,
        dst_offset);

    @autoreleasepool {
        try {
            // Buffers in this shim API are represented as "contents pointers".
            // Resolve to the backing MTLBuffer when available, otherwise treat
            // as raw CPU memory pointers.
            uint8_t* src_contents = static_cast<uint8_t*>(src_buffer);
            uint8_t* dst_contents = static_cast<uint8_t*>(dst_buffer);

            id<MTLBuffer> src_mtl_buffer = nil;
            id<MTLBuffer> dst_mtl_buffer = nil;

            auto src_it = ptr_to_mtl_buffer.find(src_buffer);
            if (src_it != ptr_to_mtl_buffer.end()) {
                src_mtl_buffer = src_it->second;
                src_contents = static_cast<uint8_t*>([src_mtl_buffer contents]);
            }

            auto dst_it = ptr_to_mtl_buffer.find(dst_buffer);
            if (dst_it != ptr_to_mtl_buffer.end()) {
                dst_mtl_buffer = dst_it->second;
                dst_contents = static_cast<uint8_t*>([dst_mtl_buffer contents]);
            }

            if (!src_contents || !dst_contents) {
                ET_LOG(Error, "aoti_torch_mps_copy_buffer: Failed to get buffer contents");
                return Error::Internal;
            }

            if (src_mtl_buffer && src_offset + data_size > [src_mtl_buffer length]) {
                ET_LOG(
                    Error,
                    "aoti_torch_mps_copy_buffer: src range out of bounds (size=%zu, offset=%zu, len=%zu)",
                    data_size,
                    src_offset,
                    static_cast<size_t>([src_mtl_buffer length]));
                return Error::InvalidArgument;
            }
            if (dst_mtl_buffer && dst_offset + data_size > [dst_mtl_buffer length]) {
                ET_LOG(
                    Error,
                    "aoti_torch_mps_copy_buffer: dst range out of bounds (size=%zu, offset=%zu, len=%zu)",
                    data_size,
                    dst_offset,
                    static_cast<size_t>([dst_mtl_buffer length]));
                return Error::InvalidArgument;
            }

            memcpy(dst_contents + dst_offset, src_contents + src_offset, data_size);

            ET_LOG(Debug, "aoti_torch_mps_copy_buffer: Copied %zu bytes from src+%zu to dst+%zu",
                   data_size, src_offset, dst_offset);
            return Error::Ok;

        } catch (const std::exception& e) {
            ET_LOG(Error, "aoti_torch_mps_copy_buffer exception: %s", e.what());
            return Error::Internal;
        } catch (...) {
            ET_LOG(Error, "aoti_torch_mps_copy_buffer: unknown exception");
            return Error::Internal;
        }
    }
}

// Shared callback function for std::function trampoline
void aoti_torch_mps_shared_callback(
    AOTIMetalKernelFunctionHandle func,
    void* user_data) {
    ET_LOG(Debug, "aoti_torch_mps_shared_callback: Called with func=%p, user_data=%p", func, user_data);

    auto* function_wrapper = static_cast<std::function<void(AOTIMetalKernelFunctionHandle)>*>(user_data);
    if (function_wrapper) {
        ET_LOG(Debug, "aoti_torch_mps_shared_callback: Calling function wrapper");
        (*function_wrapper)(func);
        ET_LOG(Debug, "aoti_torch_mps_shared_callback: Function wrapper completed");
    } else {
        ET_LOG(Error, "aoti_torch_mps_shared_callback: null function wrapper");
    }
}

// Pure C version using function pointer and user data for trampoline pattern
AOTITorchError aoti_torch_mps_run_command_block(
    AOTIMetalKernelFunctionHandle func,
    aoti_torch_mps_command_block_callback_t callback,
    void* user_data) {

    if (!func) {
        ET_LOG(Error, "aoti_torch_mps_run_command_block: null function handle");
        return Error::InvalidArgument;
    }

    if (!callback) {
        ET_LOG(Error, "aoti_torch_mps_run_command_block: null callback");
        return Error::InvalidArgument;
    }
    ET_LOG(Info, "aoti_torch_mps_run_command_block");

    ET_LOG(Debug, "aoti_torch_mps_run_command_block: Starting command block for function %p, callback %p, user_data %p",
           func, callback, user_data);

    try {
        auto* function = reinterpret_cast<ETMetalKernelFunction*>(func);
        function->clearError();
        function->runCommandBlock([callback, func, user_data]() {
            ET_LOG(Debug, "aoti_torch_mps_run_command_block: Inside lambda, calling callback");
            callback(func, user_data);
            ET_LOG(Debug, "aoti_torch_mps_run_command_block: Callback completed");
        });
        auto kernel_status =
            check_kernel_status("aoti_torch_mps_run_command_block", function);
        if (kernel_status != Error::Ok) {
            return kernel_status;
        }

        ET_LOG(Debug, "aoti_torch_mps_run_command_block: Executed command block for function %p", function);
        return Error::Ok;

    } catch (const std::exception& e) {
        ET_LOG(Error, "aoti_torch_mps_run_command_block exception: %s", e.what());
        return Error::Internal;
    } catch (...) {
        ET_LOG(Error, "aoti_torch_mps_run_command_block: unknown exception");
        return Error::Internal;
    }
}

} // extern "C"


} // namespace metal
} // namespace backends
} // namespace executorch
