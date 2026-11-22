#include "../CudaContext.h"
#include <cuda_fp16.h>
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

    // The F16 variant widens both inputs exactly to F32, evaluates the F32 formula and rounds
    // once to nearest-even. Half intermediates would round four times, and e^-x already
    // exceeds the largest finite half (65504) for x below about -11.1.
    __global__ void swiGluF16(const half* gate, const half* up, half* destination, int count) {
        const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (index < static_cast<std::size_t>(count)) {
            const float x = __half2float(gate[index]);
            const float u = __half2float(up[index]);
            destination[index] = __float2half_rn(x / (1.0f + expf(-x)) * u);
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

        case CudaEncoding::F16:
            swiGluF16<<<blocks, threads, 0, context->stream>>>(static_cast<const half*>(gate),
                static_cast<const half*>(up), static_cast<half*>(destination), count);
            break;

        default:
            // Only F32 and F16 are implemented; reject every other format before any output write.
            return cudaErrorNotSupported;
    }
    return cudaGetLastError();
}
