#include "../CudaContext.h"
#include <cuda_fp16.h>
#include <cstddef>

namespace {
// Each thread writes only its own index, so exact input/output aliasing is valid.
__global__ void addValuesF32(const float* left, const float* right, float* destination, int count) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < static_cast<std::size_t>(count)) {
        destination[index] = left[index] + right[index];
    }
}

// F16 loads and stores use two bytes per value; __hadd rounds to nearest, ties to even.
__global__ void addValuesF16(const half* left, const half* right, half* destination, int count) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < static_cast<std::size_t>(count)) {
        destination[index] = __hadd(left[index], right[index]);
    }
}
}

// count is an element count, not bytes. The managed backend handles empty tensors.
// Validate ownership, lengths and formats before dispatching from the allocation registry.
// A successful return means work was submitted on the context's stream, not completed.
MARMOTTE_EXPORT int marmotte_cuda_add(CudaContext* context, void* left, void* right, void* destination, int count) {
    const auto allocation = findBuffer(context, left, count);
    if (!allocation || !containsBuffer(context, right, count, allocation->encoding) ||
        !containsBuffer(context, destination, count, allocation->encoding)) return cudaErrorInvalidValue;
    const auto selected = cudaSetDevice(context->device);
    if (selected != cudaSuccess) return selected;

    constexpr unsigned int threads = 256;
    const auto blocks = (static_cast<unsigned int>(count) + threads - 1) / threads;
    switch (allocation->encoding) {
        case CudaEncoding::F32:
            addValuesF32<<<blocks, threads, 0, context->stream>>>(static_cast<const float*>(left),
                static_cast<const float*>(right), static_cast<float*>(destination), count);
            break;
        case CudaEncoding::F16:
            addValuesF16<<<blocks, threads, 0, context->stream>>>(static_cast<const half*>(left),
                static_cast<const half*>(right), static_cast<half*>(destination), count);
            break;
        default:
            return cudaErrorNotSupported;
    }
    return cudaGetLastError();
}
