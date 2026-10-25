#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <unordered_map>

#if defined(_WIN32)
#define MARMOTTE_EXPORT extern "C" __declspec(dllexport)
#else
#define MARMOTTE_EXPORT extern "C" __attribute__((visibility("default")))
#endif

enum class CudaEncoding { F32, F16 };

struct CudaAllocation {
    int count;
    CudaEncoding encoding;
    std::size_t bytes;
};

// Private native state shared by runtime operations and kernel launchers.
// One owner for a stream and its allocations, including finalizer-driven cleanup.
struct CudaContext {
    int device;
    cudaStream_t stream = nullptr;
    std::unordered_map<void*, CudaAllocation> buffers;
};

inline const CudaAllocation* findBuffer(CudaContext* context, void* buffer, int count) {
    if (!context) return nullptr;
    const auto entry = context->buffers.find(buffer);
    return entry != context->buffers.end() && entry->second.count == count ? &entry->second : nullptr;
}

inline bool containsBuffer(CudaContext* context, void* buffer, int count, CudaEncoding encoding) {
    const auto allocation = findBuffer(context, buffer, count);
    return allocation && allocation->encoding == encoding;
}
