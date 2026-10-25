#pragma once

#include <cuda_runtime.h>
#include <unordered_map>

#if defined(_WIN32)
#define MARMOTTE_EXPORT extern "C" __declspec(dllexport)
#else
#define MARMOTTE_EXPORT extern "C" __attribute__((visibility("default")))
#endif

// Private native state shared by runtime operations and kernel launchers.
// One owner for a stream and its allocations, including finalizer-driven cleanup.
struct CudaContext {
    int device;
    cudaStream_t stream = nullptr;
    std::unordered_map<float*, int> buffers;
};

inline bool containsBuffer(CudaContext* context, float* buffer, int count) {
    const auto entry = context->buffers.find(buffer);
    return entry != context->buffers.end() && entry->second == count;
}
