#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <unordered_map>

#if defined(_WIN32)
#define MARMOTTE_EXPORT extern "C" __declspec(dllexport)
#else
#define MARMOTTE_EXPORT extern "C" __attribute__((visibility("default")))
#endif

// Runtime tensor encodings in the same order as Core's TensorEncoding. The managed backend
// maps each case to its integer explicitly (CudaNative.fs), so this order is a shared
// contract between the two sides, never a blind cast of a foreign enum.
enum class CudaEncoding {
    F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, IQ4_NL, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K
};

inline bool storedEncodingFromInt(int value, CudaEncoding& encoding) {
    if (value < 0 || value > static_cast<int>(CudaEncoding::Q6_K)) return false;
    encoding = static_cast<CudaEncoding>(value);
    return true;
}

struct CudaAllocation {
    // Element count for execution tensors, -1 for a raw stored weight payload.
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
    // Reusable device staging for RoPE/YaRN coefficients, owned until context release.
    float* ropeCoefficients = nullptr;
    std::size_t ropeCoefficientCapacity = 0;
    // Reusable device staging for embedding lookup token IDs, owned until context release.
    int* tokenIds = nullptr;
    std::size_t tokenIdCapacity = 0;
};

// Copies a host coefficient table to reusable device staging and waits for the transfer.
// The caller can release its pinned/marshalled array immediately afterwards.
int stageRopeCoefficients(CudaContext* context, const float* coefficients, std::size_t count);

// Copies a host token ID batch to reusable device staging and waits for the transfer.
// The caller can release its pinned/marshalled array immediately afterwards.
int stageTokenIds(CudaContext* context, const int* ids, std::size_t count);

inline const CudaAllocation* findBuffer(CudaContext* context, void* buffer, int count) {
    if (!context || count <= 0) return nullptr;
    const auto entry = context->buffers.find(buffer);
    return entry != context->buffers.end() && entry->second.count == count ? &entry->second : nullptr;
}

inline bool containsBuffer(CudaContext* context, void* buffer, int count, CudaEncoding encoding) {
    const auto allocation = findBuffer(context, buffer, count);
    return allocation && allocation->encoding == encoding;
}
