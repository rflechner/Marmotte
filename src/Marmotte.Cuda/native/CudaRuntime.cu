#include "CudaContext.h"
#include <cstddef>
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

MARMOTTE_EXPORT int marmotte_cuda_allocate(CudaContext* context, int count, float** result) {
    if (!context || !result || count <= 0 ||
        static_cast<std::size_t>(count) > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
        return cudaErrorInvalidValue;
    }
    *result = nullptr;
    auto status = cudaSetDevice(context->device);
    if (status != cudaSuccess) return status;

    float* buffer = nullptr;
    const auto bytes = static_cast<std::size_t>(count) * sizeof(float);
    status = cudaMalloc(&buffer, bytes);
    if (status != cudaSuccess) return status;
    try {
        context->buffers.emplace(buffer, count);
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

MARMOTTE_EXPORT int marmotte_cuda_free(CudaContext* context, float* buffer) {
    if (!context || context->buffers.find(buffer) == context->buffers.end()) return cudaErrorInvalidValue;
    auto status = cudaSetDevice(context->device);
    if (status != cudaSuccess) return status;
    status = cudaStreamSynchronize(context->stream);
    const auto released = cudaFree(buffer);
    if (released == cudaSuccess) context->buffers.erase(buffer);
    return status == cudaSuccess ? released : status;
}

// Host pointers are valid only during P/Invoke. Wait even though the copy uses a stream.
MARMOTTE_EXPORT int marmotte_cuda_upload(CudaContext* context, float* buffer, const float* values, int count) {
    if (!context || !values || !containsBuffer(context, buffer, count)) return cudaErrorInvalidValue;
    auto status = cudaSetDevice(context->device);
    if (status != cudaSuccess) return status;
    status = cudaMemcpyAsync(buffer, values, static_cast<std::size_t>(count) * sizeof(float),
                             cudaMemcpyHostToDevice, context->stream);
    const auto completed = cudaStreamSynchronize(context->stream);
    return status == cudaSuccess ? completed : status;
}

MARMOTTE_EXPORT int marmotte_cuda_download(CudaContext* context, float* buffer, float* values, int count) {
    if (!context || !values || !containsBuffer(context, buffer, count)) return cudaErrorInvalidValue;
    auto status = cudaSetDevice(context->device);
    if (status != cudaSuccess) return status;
    status = cudaMemcpyAsync(values, buffer, static_cast<std::size_t>(count) * sizeof(float),
                             cudaMemcpyDeviceToHost, context->stream);
    const auto completed = cudaStreamSynchronize(context->stream);
    return status == cudaSuccess ? completed : status;
}

MARMOTTE_EXPORT int marmotte_cuda_synchronize(CudaContext* context) {
    if (!context) return cudaErrorInvalidValue;
    const auto selected = cudaSetDevice(context->device);
    return selected == cudaSuccess ? cudaStreamSynchronize(context->stream) : selected;
}
