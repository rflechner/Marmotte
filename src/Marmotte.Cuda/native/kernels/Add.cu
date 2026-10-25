#include "../CudaContext.h"
#include <cstddef>

namespace {
// Each thread writes only its own index, so exact input/output aliasing is valid.
__global__ void addValues(const float* left, const float* right, float* destination, int count) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < static_cast<std::size_t>(count)) {
        destination[index] = left[index] + right[index];
    }
}
}

// Host entry point: validate native storage, select the device, then queue the GPU kernel.
MARMOTTE_EXPORT int marmotte_cuda_add(CudaContext* context, float* left, float* right, float* destination, int count) {
    if (!context || !containsBuffer(context, left, count) || !containsBuffer(context, right, count) ||
        !containsBuffer(context, destination, count)) return cudaErrorInvalidValue;
    const auto selected = cudaSetDevice(context->device);
    if (selected != cudaSuccess) return selected;

    constexpr unsigned int threads = 256;
    const auto blocks = (static_cast<unsigned int>(count) + threads - 1) / threads;
    addValues<<<blocks, threads, 0, context->stream>>>(left, right, destination, count);
    return cudaGetLastError();
}
