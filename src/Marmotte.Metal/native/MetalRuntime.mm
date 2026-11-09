#include "MetalContext.h"

#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <mach-o/getsect.h>
#include <mach-o/loader.h>

namespace {
// F# reads it immediately after the failing call, on the same thread.
thread_local std::string lastError;

std::string describe(NSError* error) {
    const auto text = error.localizedDescription.UTF8String;
    return text ? text : "no description available";
}

// The shaders are a section of this library; see CMakeLists.txt.
id<MTLLibrary> loadEmbeddedLibrary(id<MTLDevice> device, NSError** error) {
    Dl_info image;
    if (dladdr(reinterpret_cast<const void*>(&loadEmbeddedLibrary), &image) == 0) return nil;

    unsigned long size = 0;
    const auto header = static_cast<const struct mach_header_64*>(image.dli_fbase);
    const auto bytes = getsectiondata(header, "__TEXT", "__metallib", &size);
    if (!bytes || size == 0) return nil;

    // The section lives as long as the loaded image: an empty destructor shares it without a copy.
    const dispatch_data_t data = dispatch_data_create(bytes, size, nullptr, ^{});
    return [device newLibraryWithData:data error:error];
}

void recordOutcome(MetalContext* context, id<MTLCommandBuffer> commandBuffer) {
    if (commandBuffer.status != MTLCommandBufferStatusError || context->deferredStatus != MetalSuccess) return;

    context->deferredStatus = commandBuffer.error.code == MTLCommandBufferErrorOutOfMemory
        ? MetalOutOfMemory
        : MetalExecutionFailure;
    context->deferredMessage = describe(commandBuffer.error);
}

void releaseCompletedWork(MetalContext* context) {
    auto& pending = context->pendingCommandBuffers;
    while (!pending.empty() && pending.front().status >= MTLCommandBufferStatusCompleted) {
        recordOutcome(context, pending.front());
        pending.pop_front();
    }
}

// Inspects every command buffer: a later one can succeed after an earlier one has failed.
int waitForSubmittedWork(MetalContext* context) {
    for (id<MTLCommandBuffer> commandBuffer : context->pendingCommandBuffers) {
        [commandBuffer waitUntilCompleted];
        recordOutcome(context, commandBuffer);
    }
    context->pendingCommandBuffers.clear();

    const auto status = context->deferredStatus;
    if (status == MetalSuccess) return MetalSuccess;

    const auto message = context->deferredMessage;
    context->deferredStatus = MetalSuccess;
    context->deferredMessage.clear();
    return failWith(status, message);
}

NSUInteger byteLength(int count, MetalEncoding encoding) {
    const NSUInteger elementBytes = encoding == MetalEncoding::F32 ? 4 : 2;
    return static_cast<NSUInteger>(count) * elementBytes;
}

// Without values, Metal returns zero-filled storage. Shared storage is visible to the GPU and
// to the host, and tracked hazards make each kernel wait for the earlier writes it depends on.
int createBuffer(MetalContext* context, const void* values, int count, MetalEncoding encoding, void** result) {
    if (!context || !result || count <= 0) return MetalInvalidArgument;
    *result = nullptr;

    const auto bytes = byteLength(count, encoding);
    if (bytes > context->device.maxBufferLength) {
        return failWith(MetalOutOfMemory, "The tensor exceeds the largest buffer of this Metal device.");
    }

    const MTLResourceOptions options = MTLResourceStorageModeShared | MTLResourceHazardTrackingModeTracked;
    id<MTLBuffer> buffer = values
        ? [context->device newBufferWithBytes:values length:bytes options:options]
        : [context->device newBufferWithLength:bytes options:options];
    if (!buffer) return failWith(MetalOutOfMemory, "Metal could not allocate the buffer.");

    const auto handle = (__bridge void*)buffer;
    context->buffers.emplace(handle, MetalAllocation{buffer, count, encoding});
    *result = handle;
    return MetalSuccess;
}

// Copies raw bytes, so every bit pattern survives, including F16 NaN payloads.
int downloadBuffer(MetalContext* context, void* buffer, void* values, int count, MetalEncoding encoding) {
    const auto allocation = findBuffer(context, buffer, count, encoding);
    if (!values || !allocation) return MetalInvalidArgument;

    @autoreleasepool {
        const auto status = waitForSubmittedWork(context);
        if (status != MetalSuccess) return status;

        std::memcpy(values, allocation->buffer.contents, byteLength(count, encoding));
        return MetalSuccess;
    }
}

int uploadBuffer(MetalContext* context, const void* values, int count, MetalEncoding encoding, void** result) {
    if (!values) return MetalInvalidArgument;

    @autoreleasepool {
        return createBuffer(context, values, count, encoding, result);
    }
}
}

int failWith(int status, const std::string& message) {
    lastError = message;
    return status;
}

int findPipeline(MetalContext* context, NSString* functionName, id<MTLComputePipelineState>* result) {
    id<MTLComputePipelineState> pipeline = context->pipelines[functionName];
    if (!pipeline) {
        id<MTLFunction> function = [context->library newFunctionWithName:functionName];
        if (!function) {
            return failWith(MetalLibraryFailure, "A kernel function is missing from the embedded Metal library.");
        }

        NSError* error = nil;
        pipeline = [context->device newComputePipelineStateWithFunction:function error:&error];
        if (!pipeline) return failWith(MetalLibraryFailure, describe(error));

        context->pipelines[functionName] = pipeline;
    }

    *result = pipeline;
    return MetalSuccess;
}

void commitCommandBuffer(MetalContext* context, id<MTLCommandBuffer> commandBuffer) {
    [commandBuffer commit];
    context->pendingCommandBuffers.push_back(commandBuffer);
    releaseCompletedWork(context);
}

// Native calls come from .NET threads that have no autorelease pool of their own, so each entry
// point that reaches Metal drains its own.

MARMOTTE_EXPORT const char* marmotte_metal_last_error() {
    return lastError.c_str();
}

MARMOTTE_EXPORT int marmotte_metal_create(MetalContext** result) {
    if (!result) return MetalInvalidArgument;
    *result = nullptr;

    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) return failWith(MetalDeviceUnavailable, "No Metal device is available.");

        // Kernels dispatch exact element counts, which needs non-uniform threadgroups.
        if (![device supportsFamily:MTLGPUFamilyApple4] && ![device supportsFamily:MTLGPUFamilyMac2]) {
            return failWith(MetalDeviceUnavailable, "This Metal device does not support non-uniform threadgroups.");
        }

        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) return failWith(MetalDeviceUnavailable, "Metal could not create a command queue.");

        NSError* error = nil;
        id<MTLLibrary> library = loadEmbeddedLibrary(device, &error);
        if (!library) {
            return failWith(MetalLibraryFailure, error ? describe(error) : "The embedded Metal library is missing.");
        }

        auto context = new MetalContext();
        context->device = device;
        context->queue = queue;
        context->library = library;
        context->pipelines = [NSMutableDictionary dictionary];
        *result = context;
        return MetalSuccess;
    }
}

// Waits so that no work outlives the context. A last failure has nobody left to report to.
MARMOTTE_EXPORT int marmotte_metal_destroy(MetalContext* context) {
    if (!context) return MetalInvalidArgument;

    @autoreleasepool {
        const auto status = waitForSubmittedWork(context);
        delete context;
        return status;
    }
}

MARMOTTE_EXPORT int marmotte_metal_allocate(MetalContext* context, int count, void** result) {
    @autoreleasepool {
        return createBuffer(context, nullptr, count, MetalEncoding::F32, result);
    }
}

MARMOTTE_EXPORT int marmotte_metal_allocate_f16(MetalContext* context, int count, void** result) {
    @autoreleasepool {
        return createBuffer(context, nullptr, count, MetalEncoding::F16, result);
    }
}

// Uploads copy the values into new storage, so no submitted work can observe the write.
MARMOTTE_EXPORT int marmotte_metal_upload(MetalContext* context, const float* values, int count, void** result) {
    return uploadBuffer(context, values, count, MetalEncoding::F32, result);
}

// The raw F16 bits. The host array uses UInt16 so that P/Invoke needs no Half marshaller.
MARMOTTE_EXPORT int marmotte_metal_upload_f16(MetalContext* context, const std::uint16_t* values, int count, void** result) {
    return uploadBuffer(context, values, count, MetalEncoding::F16, result);
}

// Does not wait: command buffers retain the resources they use until they complete.
MARMOTTE_EXPORT int marmotte_metal_free(MetalContext* context, void* buffer) {
    if (!context || context->buffers.find(buffer) == context->buffers.end()) return MetalInvalidArgument;

    @autoreleasepool {
        context->buffers.erase(buffer);
    }
    return MetalSuccess;
}

// count is an element count. The host pointer is only used during this call.
MARMOTTE_EXPORT int marmotte_metal_download(MetalContext* context, void* buffer, float* values, int count) {
    return downloadBuffer(context, buffer, values, count, MetalEncoding::F32);
}

MARMOTTE_EXPORT int marmotte_metal_download_f16(MetalContext* context, void* buffer, std::uint16_t* values, int count) {
    return downloadBuffer(context, buffer, values, count, MetalEncoding::F16);
}

MARMOTTE_EXPORT int marmotte_metal_synchronize(MetalContext* context) {
    if (!context) return MetalInvalidArgument;

    @autoreleasepool {
        return waitForSubmittedWork(context);
    }
}
