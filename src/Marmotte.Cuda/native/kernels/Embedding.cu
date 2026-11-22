#include "../CudaContext.h"
#include <cuda_fp16.h>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace {

// Stored payloads are byte sequences whose rows may start at any byte offset, so reads go
// through little-endian byte assembly instead of possibly misaligned typed loads.
__device__ __forceinline__ std::uint16_t readU16LE(const unsigned char* bytes) {
    return static_cast<std::uint16_t>(bytes[0] | (bytes[1] << 8));
}

__device__ __forceinline__ float readHalfLE(const unsigned char* bytes) {
    return __half2float(__ushort_as_half(readU16LE(bytes)));
}

__device__ __forceinline__ float readF32LE(const unsigned char* bytes) {
    const auto bits = static_cast<std::uint32_t>(bytes[0]) |
                      (static_cast<std::uint32_t>(bytes[1]) << 8) |
                      (static_cast<std::uint32_t>(bytes[2]) << 16) |
                      (static_cast<std::uint32_t>(bytes[3]) << 24);
    return __uint_as_float(bits);
}

// Decodes one value of a stored row. Only F32, F16, Q4_0 and Q8_0 are implemented (roadmap R.4);
// the host entry point rejects every other encoding before launch. Q4_0 packs two unsigned
// 4-bit codes per byte, offset by 8 and scaled by the block's F16 scale; Q8_0 stores one signed
// byte per value with the same scale. Each product is rounded explicitly, matching the CPU's
// F32 operation order without a fused multiply-add.
__device__ __forceinline__ float decodeValue(const unsigned char* row, int column, CudaEncoding encoding) {
    switch (encoding) {
        case CudaEncoding::F32:
            return readF32LE(row + 4 * column);
        case CudaEncoding::F16:
            return readHalfLE(row + 2 * column);
        case CudaEncoding::Q4_0: {
            const auto block = row + 18 * (column / 32);
            const auto packed = block[2 + column % 16];
            const auto nibble = column % 32 < 16 ? packed & 0xF : packed >> 4;
            return __fmul_rn(static_cast<float>(nibble - 8), readHalfLE(block));
        }
        case CudaEncoding::Q8_0: {
            const auto block = row + 34 * (column / 32);
            const auto quant = static_cast<std::int8_t>(block[2 + column % 32]);
            return __fmul_rn(static_cast<float>(quant), readHalfLE(block));
        }
        default:
            return 0.0f; // Unreachable: the host rejects unsupported encodings before launch.
    }
}

// One thread block decodes one requested row into its F32 destination row; each thread handles
// columns threadIdx.x, threadIdx.x + blockDim.x, ... Duplicate IDs decode independently into
// their own output rows, and the stored weights are never written.
__global__ void embeddingLookup(const unsigned char* weights, const int* tokenIds, float* destination,
                                int columns, std::size_t rowBytes, CudaEncoding encoding) {
    const auto token = blockIdx.x;
    const auto source = weights + static_cast<std::size_t>(tokenIds[token]) * rowBytes;
    auto output = destination + static_cast<std::size_t>(token) * static_cast<std::size_t>(columns);
    for (int column = static_cast<int>(threadIdx.x); column < columns; column += static_cast<int>(blockDim.x))
        output[column] = decodeValue(source, column, encoding);
}

}

// rowBytes is the stored size of one row, (columns / blockElements) * blockBytes. tokens is the
// destination row count: destination must be an F32 execution allocation of tokens * columns
// elements, and weights a stored allocation of exactly rows * rowBytes bytes in an implemented
// encoding. Host token IDs are staged to context-owned device memory before this call returns,
// so the caller may reuse its array immediately; the lookup itself is queued on the context's
// stream and may complete later. The managed backend handles empty batches and validates
// shapes, IDs and ownership first.
MARMOTTE_EXPORT int marmotte_cuda_embedding(CudaContext* context, void* weights, const int* tokenIds,
                                            void* destination, int tokens, int rows, int columns, int rowBytes) {
    if (!context || !tokenIds || tokens <= 0 || rows <= 0 || columns <= 0 || rowBytes <= 0)
        return cudaErrorInvalidValue;
    // Validate the product before multiplication, including callers outside .NET.
    if (tokens > std::numeric_limits<int>::max() / columns) return cudaErrorInvalidValue;

    const auto weightEntry = context->buffers.find(weights);
    if (weightEntry == context->buffers.end() || weightEntry->second.count != -1)
        return cudaErrorInvalidValue;
    const auto encoding = weightEntry->second.encoding;
    int blockElements, blockBytes;
    switch (encoding) {
        case CudaEncoding::F32: blockElements = 1; blockBytes = 4; break;
        case CudaEncoding::F16: blockElements = 1; blockBytes = 2; break;
        case CudaEncoding::Q4_0: blockElements = 32; blockBytes = 18; break;
        case CudaEncoding::Q8_0: blockElements = 32; blockBytes = 34; break;
        default: return cudaErrorNotSupported;
    }
    if (columns % blockElements != 0 || rowBytes != (columns / blockElements) * blockBytes)
        return cudaErrorInvalidValue;
    if (weightEntry->second.bytes != static_cast<std::size_t>(rows) * static_cast<std::size_t>(rowBytes))
        return cudaErrorInvalidValue;

    const auto values = static_cast<std::size_t>(tokens) * static_cast<std::size_t>(columns);
    const auto destinationAllocation = findBuffer(context, destination, static_cast<int>(values));
    if (!destinationAllocation || destinationAllocation->encoding != CudaEncoding::F32)
        return cudaErrorInvalidValue;

    const auto selected = cudaSetDevice(context->device);
    if (selected != cudaSuccess) return selected;
    const auto staged = stageTokenIds(context, tokenIds, static_cast<std::size_t>(tokens));
    if (staged != cudaSuccess) return staged;

    constexpr unsigned int threads = 256;
    embeddingLookup<<<static_cast<unsigned int>(tokens), threads, 0, context->stream>>>(
        static_cast<const unsigned char*>(weights), context->tokenIds,
        static_cast<float*>(destination), columns, static_cast<std::size_t>(rowBytes), encoding);
    return cudaGetLastError();
}
