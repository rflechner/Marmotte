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

// Element and coefficient positions of one pair, shared by both storage types.
// The cos/sin pair sits at token * headDimension + 2 * pair in the
// tokens * headDimension interleaved coefficient table.
struct PairLocation {
    std::size_t first;
    std::size_t second;
    std::size_t coefficient;
};

__device__ __forceinline__ PairLocation locatePair(std::size_t index, int heads,
                                                   int headDimension, int pairing) {
    const auto pairsPerHead = static_cast<std::size_t>(headDimension / 2);
    const auto tokenHead = index / pairsPerHead;
    const auto pair = index % pairsPerHead;
    const auto token = tokenHead / static_cast<std::size_t>(heads);
    const auto first = tokenHead * static_cast<std::size_t>(headDimension) +
                       (pairing == 0 ? 2 * pair : pair);
    return { first, first + (pairing == 0 ? 1 : pairsPerHead),
             token * static_cast<std::size_t>(headDimension) + 2 * pair };
}

__global__ void ropeF32(const float* input, float* destination, const float* coefficients,
                        int heads, int headDimension, int pairing, std::size_t pairCount) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= pairCount) return;

    const auto location = locatePair(index, heads, headDimension, pairing);
    const auto a = input[location.first];
    const auto b = input[location.second];
    float first, second;
    rotatePair(a, b, coefficients[location.coefficient], coefficients[location.coefficient + 1], first, second);
    destination[location.first] = first;
    destination[location.second] = second;
}

__global__ void ropeF16(const half* input, half* destination, const float* coefficients,
                        int heads, int headDimension, int pairing, std::size_t pairCount) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= pairCount) return;

    const auto location = locatePair(index, heads, headDimension, pairing);
    const float a = __half2float(input[location.first]);
    const float b = __half2float(input[location.second]);
    float first, second;
    rotatePair(a, b, coefficients[location.coefficient], coefficients[location.coefficient + 1], first, second);
    destination[location.first] = __float2half_rn(first);
    destination[location.second] = __float2half_rn(second);
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

    // Validate each product before multiplication, including callers outside .NET.
    if (heads > std::numeric_limits<int>::max() / headDimension ||
        tokens > std::numeric_limits<int>::max() / (heads * headDimension))
        return cudaErrorInvalidValue;
    const auto values = static_cast<std::size_t>(tokens) * static_cast<std::size_t>(heads) *
                        static_cast<std::size_t>(headDimension);
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
