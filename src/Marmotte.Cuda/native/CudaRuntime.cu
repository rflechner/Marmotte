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
