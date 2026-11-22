#include "../CudaContext.h"
#include <cuda_fp16.h>
#include <cstddef>
#include <limits>

namespace {

// One thread owns one pair. Both inputs are read before either output is written,
// so input == destination is safe for both adjacent and split-half layouts.
// Coefficients are prepared in F32 from double-precision angles on the host.
// Explicit rounding prevents contraction into FMA, matching the CPU's two products
// followed by one addition/subtraction.
__device__ __forceinline__ void rotatePair(float a, float b, float cosine, float sine,
                                           float& first, float& second) {
    first = __fsub_rn(__fmul_rn(a, cosine), __fmul_rn(b, sine));
    second = __fadd_rn(__fmul_rn(a, sine), __fmul_rn(b, cosine));
}

__global__ void ropeF32(const float* input, float* destination, const float* coefficients,
                        int heads, int headDimension, int pairing, std::size_t pairCount) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= pairCount) return;

    const auto pairsPerHead = static_cast<std::size_t>(headDimension / 2);
    const auto tokenHead = index / pairsPerHead;
    const auto pair = index % pairsPerHead;
    const auto token = tokenHead / static_cast<std::size_t>(heads);
    const auto base = tokenHead * static_cast<std::size_t>(headDimension);
    const auto firstIndex = base + (pairing == 0 ? 2 * pair : pair);
    const auto secondIndex = firstIndex + (pairing == 0 ? 1 : pairsPerHead);
    const auto coefficientIndex = 2 * (token * pairsPerHead + pair);

    const auto a = input[firstIndex];
    const auto b = input[secondIndex];
    float first, second;
    rotatePair(a, b, coefficients[coefficientIndex], coefficients[coefficientIndex + 1], first, second);
    destination[firstIndex] = first;
    destination[secondIndex] = second;
}

__global__ void ropeF16(const half* input, half* destination, const float* coefficients,
                        int heads, int headDimension, int pairing, std::size_t pairCount) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= pairCount) return;

    const auto pairsPerHead = static_cast<std::size_t>(headDimension / 2);
    const auto tokenHead = index / pairsPerHead;
    const auto pair = index % pairsPerHead;
    const auto token = tokenHead / static_cast<std::size_t>(heads);
    const auto base = tokenHead * static_cast<std::size_t>(headDimension);
    const auto firstIndex = base + (pairing == 0 ? 2 * pair : pair);
    const auto secondIndex = firstIndex + (pairing == 0 ? 1 : pairsPerHead);
    const auto coefficientIndex = 2 * (token * pairsPerHead + pair);

    const float a = __half2float(input[firstIndex]);
    const float b = __half2float(input[secondIndex]);
    float first, second;
    rotatePair(a, b, coefficients[coefficientIndex], coefficients[coefficientIndex + 1], first, second);
    destination[firstIndex] = __float2half_rn(first);
    destination[secondIndex] = __float2half_rn(second);
}
}

// tokens * heads * headDimension is the execution tensor element count.
// coefficients contains tokens * headDimension F32 values: interleaved cos/sin
// for each pair of each token (shared by every head). pairing: 0 adjacent, 1 split.
// Host staging completes before return to release the marshalled managed array;
// the rotation itself is queued on the context's stream and may complete later.
// The managed backend handles empty tensors and checks shape/positions first.
MARMOTTE_EXPORT int marmotte_cuda_rope(CudaContext* context, void* input, void* destination,
                                      const float* coefficients, int tokens, int heads,
                                      int headDimension, int pairing) {
    if (!context || !coefficients || tokens <= 0 || heads <= 0 || headDimension <= 0 ||
        headDimension % 2 != 0 || (pairing != 0 && pairing != 1)) return cudaErrorInvalidValue;

    const auto values = static_cast<std::size_t>(tokens) * static_cast<std::size_t>(heads) *
                        static_cast<std::size_t>(headDimension);
    if (values > static_cast<std::size_t>(std::numeric_limits<int>::max())) return cudaErrorInvalidValue;
    const auto allocation = findBuffer(context, input, static_cast<int>(values));
    if (!allocation || !containsBuffer(context, destination, static_cast<int>(values), allocation->encoding))
        return cudaErrorInvalidValue;
    if (allocation->encoding != CudaEncoding::F32 && allocation->encoding != CudaEncoding::F16)
        return cudaErrorNotSupported;

    const auto selected = cudaSetDevice(context->device);
    if (selected != cudaSuccess) return selected;
    const auto staged = stageRopeCoefficients(context, coefficients,
                                             static_cast<std::size_t>(tokens) * headDimension);
    if (staged != cudaSuccess) return staged;

    const auto pairCount = values / 2;
    constexpr unsigned int threads = 256;
    const auto blocks = static_cast<unsigned int>((pairCount + threads - 1) / threads);
    if (allocation->encoding == CudaEncoding::F32) {
        ropeF32<<<blocks, threads, 0, context->stream>>>(
            static_cast<const float*>(input), static_cast<float*>(destination),
            context->ropeCoefficients, heads, headDimension, pairing, pairCount);
    } else {
        ropeF16<<<blocks, threads, 0, context->stream>>>(
            static_cast<const half*>(input), static_cast<half*>(destination),
            context->ropeCoefficients, heads, headDimension, pairing, pairCount);
    }
    return cudaGetLastError();
}
