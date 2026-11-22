#include "CudaContext.h"
#include <cuda_fp16.h>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>

MARMOTTE_EXPORT const char* marmotte_cuda_error(int status) {
    return cudaGetErrorString(static_cast<cudaError_t>(status));
}

MARMOTTE_EXPORT int marmotte_cuda_create(int device, CudaContext** result) {
    if (!result) return cudaErrorInvalidValue;
    *result = nullptr;
    const auto selected = cudaSetDevice(device);
    if (selected != cudaSuccess) return selected;

    auto context = new (std::nothrow) CudaContext{device};
    if (!context) return cudaErrorMemoryAllocation;
    const auto status = cudaStreamCreateWithFlags(&context->stream, cudaStreamNonBlocking);
    if (status != cudaSuccess) {
        delete context;
        return status;
    }
    *result = context;
    return cudaSuccess;
}

// Always attempt every release, even after a failed asynchronous operation.
MARMOTTE_EXPORT int marmotte_cuda_destroy(CudaContext* context) {
    if (!context) return cudaSuccess;
    auto status = cudaSetDevice(context->device);
    if (status == cudaSuccess) {
        status = cudaStreamSynchronize(context->stream);
        for (const auto& entry : context->buffers) {
            const auto released = cudaFree(entry.first);
            if (status == cudaSuccess) status = released;
        }
        if (context->ropeCoefficients) {
            const auto released = cudaFree(context->ropeCoefficients);
            if (status == cudaSuccess) status = released;
        }
        if (context->tokenIds) {
            const auto released = cudaFree(context->tokenIds);
            if (status == cudaSuccess) status = released;
        }
        const auto destroyed = cudaStreamDestroy(context->stream);
        if (status == cudaSuccess) status = destroyed;
    }
    delete context;
    return status;
}

namespace {
static_assert(sizeof(half) == 2 && sizeof(float) == 4 && sizeof(std::uint16_t) == 2);

int allocateBuffer(CudaContext* context, int count, CudaEncoding encoding, void** result) {
    const auto elementBytes = encoding == CudaEncoding::F16 ? sizeof(half) : sizeof(float);
    if (!context || !result || count <= 0 ||
        static_cast<std::size_t>(count) > std::numeric_limits<std::size_t>::max() / elementBytes) {
        return cudaErrorInvalidValue;
    }
    *result = nullptr;
    auto status = cudaSetDevice(context->device);
    if (status != cudaSuccess) return status;

    void* buffer = nullptr;
    const auto bytes = static_cast<std::size_t>(count) * elementBytes;
    status = cudaMalloc(&buffer, bytes);
    if (status != cudaSuccess) return status;
    try {
        context->buffers.emplace(buffer, CudaAllocation{count, encoding, bytes});
    } catch (const std::bad_alloc&) {
        cudaFree(buffer);
        return cudaErrorMemoryAllocation;
    }

    status = cudaMemsetAsync(buffer, 0, bytes, context->stream);
    if (status != cudaSuccess) {
        // Retain a failed release for a final cleanup attempt when the context closes.
        if (cudaFree(buffer) == cudaSuccess) context->buffers.erase(buffer);
        return status;
    }
    *result = buffer;
    return cudaSuccess;
}
}

MARMOTTE_EXPORT int marmotte_cuda_allocate(CudaContext* context, int count, void** result) {
    return allocateBuffer(context, count, CudaEncoding::F32, result);
}

MARMOTTE_EXPORT int marmotte_cuda_allocate_f16(CudaContext* context, int count, void** result) {
    return allocateBuffer(context, count, CudaEncoding::F16, result);
}

// A stored payload is never interpreted or zeroed: upload fills it before publication.
MARMOTTE_EXPORT int marmotte_cuda_allocate_stored(CudaContext* context, int encodingValue, long long bytes, void** result) {
    if (result) *result = nullptr;
    CudaEncoding encoding;
    if (!context || !result || bytes <= 0 ||
        static_cast<unsigned long long>(bytes) > std::numeric_limits<std::size_t>::max() ||
        !storedEncodingFromInt(encodingValue, encoding)) return cudaErrorInvalidValue;

    auto status = cudaSetDevice(context->device);
    if (status != cudaSuccess) return status;
    void* buffer = nullptr;
    status = cudaMalloc(&buffer, static_cast<std::size_t>(bytes));
    if (status != cudaSuccess) return status;
    try {
        context->buffers.emplace(buffer, CudaAllocation{-1, encoding, static_cast<std::size_t>(bytes)});
    } catch (const std::bad_alloc&) {
        cudaFree(buffer);
        return cudaErrorMemoryAllocation;
    }
    *result = buffer;
    return cudaSuccess;
}

namespace {
const CudaAllocation* storedRange(CudaContext* context, void* buffer, long long offset, long long bytes) {
    if (!context || offset < 0 || bytes <= 0) return nullptr;
    const auto entry = context->buffers.find(buffer);
    if (entry == context->buffers.end() || entry->second.count != -1) return nullptr;
    const auto& allocation = entry->second;
    if (static_cast<unsigned long long>(offset) > allocation.bytes ||
        static_cast<unsigned long long>(bytes) > allocation.bytes - static_cast<std::size_t>(offset)) return nullptr;
    return &allocation;
}
}

// Offsets/counts are bytes, not elements. A range may end inside a quantization block.
// The host pointer is valid only during P/Invoke; wait before reusing the bounded staging array.
MARMOTTE_EXPORT int marmotte_cuda_upload_stored_range(CudaContext* context, void* buffer, long long offset,
                                                    const void* payload, long long byteCount) {
    if (!payload || !storedRange(context, buffer, offset, byteCount)) return cudaErrorInvalidValue;
    auto status = cudaSetDevice(context->device);
    if (status != cudaSuccess) return status;
    auto destination = static_cast<unsigned char*>(buffer) + static_cast<std::size_t>(offset);
    status = cudaMemcpyAsync(destination, payload, static_cast<std::size_t>(byteCount),
                             cudaMemcpyHostToDevice, context->stream);
    const auto completed = cudaStreamSynchronize(context->stream);
    return status == cudaSuccess ? completed : status;
}

// Exact raw readback and allocation accounting support storage diagnostics without decoding.
MARMOTTE_EXPORT int marmotte_cuda_download_stored(CudaContext* context, void* buffer, void* payload, long long bytes) {
    const auto allocation = storedRange(context, buffer, 0, bytes);
    if (!payload || !allocation || allocation->bytes != static_cast<unsigned long long>(bytes)) return cudaErrorInvalidValue;
    auto status = cudaSetDevice(context->device);
    if (status != cudaSuccess) return status;
    status = cudaMemcpyAsync(payload, buffer, allocation->bytes, cudaMemcpyDeviceToHost, context->stream);
    const auto completed = cudaStreamSynchronize(context->stream);
    return status == cudaSuccess ? completed : status;
}

MARMOTTE_EXPORT int marmotte_cuda_allocated_bytes(CudaContext* context, long long* bytes) {
    if (!context || !bytes) return cudaErrorInvalidValue;
    unsigned long long total = 0;
    for (const auto& entry : context->buffers) {
        if (entry.second.bytes > static_cast<unsigned long long>(std::numeric_limits<long long>::max()) - total)
            return cudaErrorInvalidValue;
        total += entry.second.bytes;
    }
    const auto addStaging = [&total](std::size_t elements, std::size_t elementSize) {
        if (elements > (static_cast<unsigned long long>(std::numeric_limits<long long>::max()) - total) / elementSize)
            return false;
        total += elements * elementSize;
        return true;
    };
    if (!addStaging(context->ropeCoefficientCapacity, sizeof(float)) ||
        !addStaging(context->tokenIdCapacity, sizeof(int))) return cudaErrorInvalidValue;
    *bytes = static_cast<long long>(total);
    return cudaSuccess;
}

namespace {

// Host arrays from .NET are only valid for this P/Invoke call. Copy into reusable
// context-owned staging and wait for the H2D transfer before returning, so the caller may
// reuse its array immediately; the kernel that reads the staging is submitted asynchronously.
// Old launches may still read the previous buffer, so growing the staging synchronizes the
// stream before replacing it. The staging is released on context destruction.
template <typename T>
int stageHostBuffer(CudaContext* context, T*& staging, std::size_t& capacity,
                    const T* source, std::size_t count) {
    if (!context || !source || count == 0 ||
        count > std::numeric_limits<std::size_t>::max() / sizeof(T)) return cudaErrorInvalidValue;
    auto status = cudaSetDevice(context->device);
    if (status != cudaSuccess) return status;

    if (count > capacity) {
        status = cudaStreamSynchronize(context->stream);
        if (status != cudaSuccess) return status;
        T* replacement = nullptr;
        status = cudaMalloc(&replacement, count * sizeof(T));
        if (status != cudaSuccess) return status;
        if (staging) {
            const auto released = cudaFree(staging);
            if (released != cudaSuccess) {
                cudaFree(replacement);
                return released;
            }
        }
        staging = replacement;
        capacity = count;
    }
    status = cudaMemcpyAsync(staging, source, count * sizeof(T), cudaMemcpyHostToDevice, context->stream);
    const auto completed = cudaStreamSynchronize(context->stream);
    return status == cudaSuccess ? completed : status;
}

}

int stageRopeCoefficients(CudaContext* context, const float* coefficients, std::size_t count) {
    return stageHostBuffer(context, context->ropeCoefficients, context->ropeCoefficientCapacity,
                           coefficients, count);
}

int stageTokenIds(CudaContext* context, const int* ids, std::size_t count) {
    return stageHostBuffer(context, context->tokenIds, context->tokenIdCapacity, ids, count);
}

MARMOTTE_EXPORT int marmotte_cuda_free(CudaContext* context, void* buffer) {
    if (!context || context->buffers.find(buffer) == context->buffers.end()) return cudaErrorInvalidValue;
    auto status = cudaSetDevice(context->device);
    if (status != cudaSuccess) return status;
    status = cudaStreamSynchronize(context->stream);
    const auto released = cudaFree(buffer);
    if (released == cudaSuccess) context->buffers.erase(buffer);
    return status == cudaSuccess ? released : status;
}

namespace {
// Host pointers are valid only during P/Invoke. Wait even though the copy uses a stream.
int uploadBuffer(CudaContext* context, void* buffer, const void* values, int count, CudaEncoding encoding) {
    const auto allocation = findBuffer(context, buffer, count);
    if (!values || !allocation || allocation->encoding != encoding) return cudaErrorInvalidValue;
    auto status = cudaSetDevice(context->device);
    if (status != cudaSuccess) return status;
    status = cudaMemcpyAsync(buffer, values, allocation->bytes,
                             cudaMemcpyHostToDevice, context->stream);
    const auto completed = cudaStreamSynchronize(context->stream);
    return status == cudaSuccess ? completed : status;
}

int downloadBuffer(CudaContext* context, void* buffer, void* values, int count, CudaEncoding encoding) {
    const auto allocation = findBuffer(context, buffer, count);
    if (!values || !allocation || allocation->encoding != encoding) return cudaErrorInvalidValue;
    auto status = cudaSetDevice(context->device);
    if (status != cudaSuccess) return status;
    status = cudaMemcpyAsync(values, buffer, allocation->bytes,
                             cudaMemcpyDeviceToHost, context->stream);
    const auto completed = cudaStreamSynchronize(context->stream);
    return status == cudaSuccess ? completed : status;
}
}

MARMOTTE_EXPORT int marmotte_cuda_upload(CudaContext* context, void* buffer, const float* values, int count) {
    return uploadBuffer(context, buffer, values, count, CudaEncoding::F32);
}

MARMOTTE_EXPORT int marmotte_cuda_download(CudaContext* context, void* buffer, float* values, int count) {
    return downloadBuffer(context, buffer, values, count, CudaEncoding::F32);
}

// Transfer the raw F16 bits. The host staging array uses UInt16 so P/Invoke needs no Half marshaller.
MARMOTTE_EXPORT int marmotte_cuda_upload_f16(CudaContext* context, void* buffer, const std::uint16_t* values, int count) {
    return uploadBuffer(context, buffer, values, count, CudaEncoding::F16);
}

MARMOTTE_EXPORT int marmotte_cuda_download_f16(CudaContext* context, void* buffer, std::uint16_t* values, int count) {
    return downloadBuffer(context, buffer, values, count, CudaEncoding::F16);
}

MARMOTTE_EXPORT int marmotte_cuda_synchronize(CudaContext* context) {
    if (!context) return cudaErrorInvalidValue;
    const auto selected = cudaSetDevice(context->device);
    return selected == cudaSuccess ? cudaStreamSynchronize(context->stream) : selected;
}
