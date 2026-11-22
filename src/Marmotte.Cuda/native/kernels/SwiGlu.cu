#include "../CudaContext.h"
#include <cmath>
#include <cstddef>

namespace {

    // Each thread writes only its own index, so exact input/output aliasing is valid.
    // silu(x) = x / (1 + e^-x), with no fast math: expf is the single-precision device
    // exponential, so results stay within a few ULP of the CPU reference.
    __global__ void swiGluF32(const float* gate, const float* up, float* destination, int count) {

        const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (index < static_cast<std::size_t>(count)) {

            const auto x = gate[index];
            destination[index] = x / (1.0f + expf(-x)) * up[index];
        }
    }
}

// count is an element count, not bytes. The managed backend handles empty tensors.
// Validate ownership, lengths and formats before dispatching from the allocation registry.
// A successful return means work was submitted on the context's stream, not completed.
MARMOTTE_EXPORT int marmotte_cuda_swiglu(CudaContext* context, void* gate, void* up, void* destination, int count) {

    const auto allocation = findBuffer(context, gate, count);
    if (!allocation || !containsBuffer(context, up, count, allocation->encoding) ||
        !containsBuffer(context, destination, count, allocation->encoding)) return cudaErrorInvalidValue;
    const auto selected = cudaSetDevice(context->device);
    if (selected != cudaSuccess) return selected;

    constexpr unsigned int threads = 256;
    const auto blocks = (static_cast<unsigned int>(count) + threads - 1) / threads;
    switch (allocation->encoding) {

        case CudaEncoding::F32:
            swiGluF32<<<blocks, threads, 0, context->stream>>>(static_cast<const float*>(gate),
                static_cast<const float*>(up), static_cast<float*>(destination), count);
            break;

        default:
            // Only F32 is implemented; reject every other format before any output write.
            return cudaErrorNotSupported;
    }
    return cudaGetLastError();
}
