#include "../CudaContext.h"
#include <cuda_fp16.h>
#include <cmath>
#include <cstddef>
#include <limits>

namespace {

// One block per row of `columns` values. The sum of squares accumulates in double
// precision: rows of thousands of values keep F32 accuracy, and squares beyond the
// F32 range, such as that of 1e20, do not overflow. The scale 1 / sqrt(mean + epsilon)
// is computed in double and rounded once to F32, then applied as value * scale * weight
// with two F32 roundings, matching the CPU reference.
// The block reduction completes before any write, so destination may be input; each
// column is read from weight by the same thread that writes its destination.
__global__ void rmsNormF32(const float* input, const float* weight, float epsilon,
                           float* destination, int rows, int columns) {

    const auto row = static_cast<std::size_t>(blockIdx.x);
    if (row >= static_cast<std::size_t>(rows)) return;
    const auto base = row * static_cast<std::size_t>(columns);
    const auto width = static_cast<std::size_t>(columns);

    double sum = 0.0;
    for (auto column = static_cast<std::size_t>(threadIdx.x); column < width; column += blockDim.x) {

        const auto value = static_cast<double>(input[base + column]);
        sum += value * value;
    }

    __shared__ double partials[256];
    partials[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {

        if (threadIdx.x < stride) partials[threadIdx.x] += partials[threadIdx.x + stride];
        __syncthreads();
    }

    const auto mean = partials[0] / static_cast<double>(columns);
    const auto scale = static_cast<float>(1.0 / std::sqrt(mean + static_cast<double>(epsilon)));
    for (auto column = static_cast<std::size_t>(threadIdx.x); column < width; column += blockDim.x) {

        destination[base + column] = input[base + column] * scale * weight[column];
    }
}

// The F16 variant widens both inputs exactly to F32 with __half2float, runs the same
// double-precision reduction and F32 scale, and rounds the final product once to
// nearest-even. Half intermediates would round after every operation, and the largest
// half (65504) already loses accuracy when squared in F32.
__global__ void rmsNormF16(const half* input, const half* weight, float epsilon,
                           half* destination, int rows, int columns) {

    const auto row = static_cast<std::size_t>(blockIdx.x);
    if (row >= static_cast<std::size_t>(rows)) return;
    const auto base = row * static_cast<std::size_t>(columns);
    const auto width = static_cast<std::size_t>(columns);

    double sum = 0.0;
    for (auto column = static_cast<std::size_t>(threadIdx.x); column < width; column += blockDim.x) {

        const auto value = static_cast<double>(__half2float(input[base + column]));
        sum += value * value;
    }

    __shared__ double partials[256];
    partials[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {

        if (threadIdx.x < stride) partials[threadIdx.x] += partials[threadIdx.x + stride];
        __syncthreads();
    }

    const auto mean = partials[0] / static_cast<double>(columns);
    const auto scale = static_cast<float>(1.0 / std::sqrt(mean + static_cast<double>(epsilon)));
    for (auto column = static_cast<std::size_t>(threadIdx.x); column < width; column += blockDim.x) {

        destination[base + column] = __float2half_rn(__half2float(input[base + column]) * scale * __half2float(weight[column]));
    }
}
}

// rows and columns count elements, not bytes; rows * columns is the input and destination
// length and columns the weight length. The managed backend handles empty tensors and
// validates shapes and epsilon. Validate ownership, lengths and formats before dispatching
// from the allocation registry. A successful return means work was submitted on the
// context's stream, not completed.
MARMOTTE_EXPORT int marmotte_cuda_rmsnorm(CudaContext* context, void* input, void* weight, float epsilon,
                                          void* destination, int rows, int columns) {

    if (rows <= 0 || columns <= 0 || rows > std::numeric_limits<int>::max() / columns) return cudaErrorInvalidValue;
    const auto count = rows * columns;
    const auto allocation = findBuffer(context, input, count);
    if (!allocation || !containsBuffer(context, weight, columns, allocation->encoding) ||
        !containsBuffer(context, destination, count, allocation->encoding)) return cudaErrorInvalidValue;
    const auto selected = cudaSetDevice(context->device);
    if (selected != cudaSuccess) return selected;

    constexpr unsigned int threads = 256;
    switch (allocation->encoding) {

        case CudaEncoding::F32:
            rmsNormF32<<<static_cast<unsigned int>(rows), threads, 0, context->stream>>>(
                static_cast<const float*>(input), static_cast<const float*>(weight), epsilon,
                static_cast<float*>(destination), rows, columns);
            break;

        case CudaEncoding::F16:
            rmsNormF16<<<static_cast<unsigned int>(rows), threads, 0, context->stream>>>(
                static_cast<const half*>(input), static_cast<const half*>(weight), epsilon,
                static_cast<half*>(destination), rows, columns);
            break;

        default:
            // Only F32 and F16 are implemented; reject every other format before any output write.
            return cudaErrorNotSupported;
    }
    return cudaGetLastError();
}
