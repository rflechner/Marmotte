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

// Decodes one value of a stored row. Only F32 and F16 are implemented (roadmap S.1/S.2);
// the host entry points reject every other encoding before launch.
__device__ __forceinline__ float decodeStoredValue(const unsigned char* row, int column, CudaEncoding encoding) {
    switch (encoding) {
        case CudaEncoding::F32:
            return readF32LE(row + 4 * column);
        case CudaEncoding::F16:
            return readHalfLE(row + 2 * column);
        default:
            return 0.0f; // Unreachable: the host rejects unsupported encodings before launch.
    }
}

constexpr unsigned int MATMUL_THREADS = 256;
constexpr int WARPS_PER_BLOCK = static_cast<int>(MATMUL_THREADS / 32);
// Each block of the batched product decodes this many weight rows, one after another.
constexpr int ROWS_PER_BLOCK = 4;
// Column tile of one decoded weight row held in shared memory, in F32 values (8 KiB).
constexpr int COLUMN_CHUNK = 2048;

// Reference matrix-vector product: one block per weight row, decoding the row on the fly.
// Each thread accumulates its strided columns in double precision, so every product of two
// F32 values is exact and only the final sum rounds; the warp and block reductions keep the
// same precision and round once to F32 on write, matching the CPU reference's tolerance.
__global__ void matVecKernel(const unsigned char* weights, const float* input, float* destination,
                             int columns, std::size_t rowBytes, CudaEncoding encoding) {
    const int row = blockIdx.x;
    const unsigned char* rowPtr = weights + static_cast<std::size_t>(row) * rowBytes;
    double sum = 0.0;
    for (int column = threadIdx.x; column < columns; column += blockDim.x)
        sum += static_cast<double>(decodeStoredValue(rowPtr, column, encoding)) * static_cast<double>(input[column]);

    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    __shared__ double warpSums[WARPS_PER_BLOCK];
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int warp = threadIdx.x >> 5;
    if (lane == 0) warpSums[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        double total = lane < WARPS_PER_BLOCK ? warpSums[lane] : 0.0;
        for (int offset = WARPS_PER_BLOCK / 2; offset > 0; offset >>= 1)
            total += __shfl_down_sync(0xffffffffu, total, offset);
        if (lane == 0) destination[row] = static_cast<float>(total);
    }
}

// Batched matrix product: each block decodes its tile of weight rows once into shared memory,
// in column chunks, and shares every decoded row across the whole token slice of the launch.
// Thread t accumulates token t's dot product in partials[t], in double precision as above.
// The host slices larger batches so that tokens <= blockDim.x, so each row is decoded once
// per slice instead of once per token as in the reference path.
__global__ void matMulBatchedKernel(const unsigned char* weights, const float* input, float* destination,
                                    int tokens, int rows, int columns, std::size_t rowBytes, CudaEncoding encoding) {
    __shared__ float sharedRow[COLUMN_CHUNK];
    __shared__ double partials[MATMUL_THREADS];

    const int firstRow = blockIdx.x * ROWS_PER_BLOCK;
    const int rowCount = rows - firstRow < ROWS_PER_BLOCK ? rows - firstRow : ROWS_PER_BLOCK;
    const int token = threadIdx.x;

    for (int tile = 0; tile < rowCount; ++tile) {
        const int row = firstRow + tile;
        const unsigned char* rowPtr = weights + static_cast<std::size_t>(row) * rowBytes;
        if (token < tokens) partials[token] = 0.0;
        __syncthreads();
        for (int chunkBase = 0; chunkBase < columns; chunkBase += COLUMN_CHUNK) {
            const int chunkLength = columns - chunkBase < COLUMN_CHUNK ? columns - chunkBase : COLUMN_CHUNK;
            for (int column = threadIdx.x; column < chunkLength; column += blockDim.x)
                sharedRow[column] = decodeStoredValue(rowPtr, chunkBase + column, encoding);
            __syncthreads();
            if (token < tokens) {
                // Thread t owns token t: it alone walks every column of the chunk, so the
                // decoded row in shared memory is shared by the whole token slice.
                const float* inputRow = input + static_cast<std::size_t>(token) * columns + chunkBase;
                for (int column = 0; column < chunkLength; ++column)
                    partials[token] += static_cast<double>(sharedRow[column]) * static_cast<double>(inputRow[column]);
            }
            __syncthreads();
        }
        if (token < tokens)
            destination[static_cast<std::size_t>(token) * rows + row] = static_cast<float>(partials[token]);
    }
}

// Revalidates the stored weight allocation: F32/F16 only, rowBytes consistent with the
// encoding's block layout, and a payload of exactly rows * rowBytes bytes.
int storedWeightMatrix(CudaContext* context, void* weights, CudaEncoding& encoding,
                       int rows, int columns, int rowBytes) {
    if (!context || !weights || rows <= 0 || columns <= 0 || rowBytes <= 0) return cudaErrorInvalidValue;
    const auto entry = context->buffers.find(weights);
    if (entry == context->buffers.end() || entry->second.count != -1) return cudaErrorInvalidValue;
    const auto storedEncoding = entry->second.encoding;
    int blockElements, blockBytes;
    switch (storedEncoding) {
        case CudaEncoding::F32: blockElements = 1; blockBytes = 4; break;
        case CudaEncoding::F16: blockElements = 1; blockBytes = 2; break;
        default: return cudaErrorNotSupported;
    }
    if (columns % blockElements != 0 || rowBytes != (columns / blockElements) * blockBytes)
        return cudaErrorInvalidValue;
    if (entry->second.bytes != static_cast<std::size_t>(rows) * static_cast<std::size_t>(rowBytes))
        return cudaErrorInvalidValue;
    encoding = storedEncoding;
    return cudaSuccess;
}

// An execution operand must be an F32 allocation of exactly count elements.
int f32Vector(CudaContext* context, void* buffer, long long count) {
    if (!context || !buffer || count <= 0 || count > std::numeric_limits<int>::max())
        return cudaErrorInvalidValue;
    return containsBuffer(context, buffer, static_cast<int>(count), CudaEncoding::F32)
               ? cudaSuccess
               : cudaErrorInvalidValue;
}

} // namespace

// destination[r] = sum over c of decode(weights[r, c]) * input[c], with input an F32 vector
// of columns elements and destination an F32 vector of rows elements. rowBytes is the stored
// size of one row, (columns / blockElements) * blockBytes. weights must be a stored allocation
// of exactly rows * rowBytes bytes in F32 or F16; every other encoding returns
// cudaErrorNotSupported before any launch, and input == destination is rejected because each
// output reads the whole input. The managed backend handles validation and empty products;
// this entry point launches on the context's stream and returns without waiting.
MARMOTTE_EXPORT int marmotte_cuda_matvec(CudaContext* context, void* weights, void* input, void* destination,
                                         int rows, int columns, int rowBytes) {
    if (!context || !input || !destination || input == destination) return cudaErrorInvalidValue;
    CudaEncoding encoding;
    auto status = storedWeightMatrix(context, weights, encoding, rows, columns, rowBytes);
    if (status != cudaSuccess) return status;
    status = f32Vector(context, input, columns);
    if (status != cudaSuccess) return status;
    status = f32Vector(context, destination, rows);
    if (status != cudaSuccess) return status;

    const auto selected = cudaSetDevice(context->device);
    if (selected != cudaSuccess) return selected;
    matVecKernel<<<static_cast<unsigned int>(rows), MATMUL_THREADS, 0, context->stream>>>(
        static_cast<const unsigned char*>(weights), static_cast<const float*>(input),
        static_cast<float*>(destination), columns, static_cast<std::size_t>(rowBytes), encoding);
    return cudaGetLastError();
}

// destination[t, r] = sum over c of decode(weights[r, c]) * input[t, c], with input an F32
// matrix of tokens * columns elements and destination an F32 matrix of tokens * rows
// elements, both row-major. A single-token batch launches the reference matrix-vector kernel;
// larger batches are sliced into groups of at most MATMUL_THREADS tokens, each decoded once
// per weight row into shared memory by the batched kernel. tokens must be positive; the
// managed backend handles the empty batch as a no-op. Shapes, encodings, ownership and the
// input != destination rule are revalidated here before any write.
MARMOTTE_EXPORT int marmotte_cuda_matmul(CudaContext* context, void* weights, void* input, void* destination,
                                         int tokens, int rows, int columns, int rowBytes) {
    if (!context || !input || !destination || input == destination || tokens <= 0) return cudaErrorInvalidValue;
    CudaEncoding encoding;
    auto status = storedWeightMatrix(context, weights, encoding, rows, columns, rowBytes);
    if (status != cudaSuccess) return status;
    status = f32Vector(context, input, static_cast<long long>(tokens) * columns);
    if (status != cudaSuccess) return status;
    status = f32Vector(context, destination, static_cast<long long>(tokens) * rows);
    if (status != cudaSuccess) return status;

    const auto selected = cudaSetDevice(context->device);
    if (selected != cudaSuccess) return selected;

    if (tokens == 1) {
        matVecKernel<<<static_cast<unsigned int>(rows), MATMUL_THREADS, 0, context->stream>>>(
            static_cast<const unsigned char*>(weights), static_cast<const float*>(input),
            static_cast<float*>(destination), columns, static_cast<std::size_t>(rowBytes), encoding);
        return cudaGetLastError();
    }

    const auto blocks = static_cast<unsigned int>(
        (static_cast<long long>(rows) + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK);
    for (int tokenBase = 0; tokenBase < tokens; tokenBase += static_cast<int>(MATMUL_THREADS)) {
        const int slice = tokens - tokenBase < static_cast<int>(MATMUL_THREADS)
                              ? tokens - tokenBase
                              : static_cast<int>(MATMUL_THREADS);
        const float* inputSlice = static_cast<const float*>(input) + static_cast<std::size_t>(tokenBase) * columns;
        float* destinationSlice = static_cast<float*>(destination) + static_cast<std::size_t>(tokenBase) * rows;
        matMulBatchedKernel<<<blocks, MATMUL_THREADS, 0, context->stream>>>(
            static_cast<const unsigned char*>(weights), inputSlice, destinationSlice,
            slice, rows, columns, static_cast<std::size_t>(rowBytes), encoding);
        status = cudaGetLastError();
        if (status != cudaSuccess) return status;
    }
    return cudaSuccess;
}
