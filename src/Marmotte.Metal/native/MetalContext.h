#pragma once

#import <Metal/Metal.h>

#include <deque>
#include <string>
#include <unordered_map>

#define MARMOTTE_EXPORT extern "C" __attribute__((visibility("default")))

// Status codes of the private C ABI. Keep in sync with MetalNative.fs.
enum MetalStatus : int {
    MetalSuccess = 0,
    MetalInvalidArgument = 1,
    MetalOutOfMemory = 2,
    MetalDeviceUnavailable = 3,
    MetalLibraryFailure = 4,
    MetalExecutionFailure = 5,
};

enum class MetalEncoding { F32, F16 };

struct MetalAllocation {
    id<MTLBuffer> buffer;
    int count;
    MetalEncoding encoding;
};

// Private native state shared by runtime operations and kernel entry points.
// One owner for a queue and its allocations, including finalizer-driven cleanup.
// Concurrent calls on one context are not supported.
struct MetalContext {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary> library;
    NSMutableDictionary<NSString*, id<MTLComputePipelineState>>* pipelines;

    // Keyed by the opaque handle given to F#, which never dereferences it.
    std::unordered_map<void*, MetalAllocation> buffers;

    // Committed command buffers, oldest first, whose outcome has not been read yet.
    std::deque<id<MTLCommandBuffer>> pendingCommandBuffers;

    // First failure seen among completed command buffers, until a wait reports it.
    int deferredStatus = MetalSuccess;
    std::string deferredMessage;
};

inline const MetalAllocation* findBuffer(MetalContext* context, void* buffer, int count) {
    if (!context) return nullptr;
    const auto entry = context->buffers.find(buffer);
    return entry != context->buffers.end() && entry->second.count == count ? &entry->second : nullptr;
}

inline const MetalAllocation* findBuffer(MetalContext* context, void* buffer, int count, MetalEncoding encoding) {
    const auto allocation = findBuffer(context, buffer, count);
    return allocation && allocation->encoding == encoding ? allocation : nullptr;
}

// Records a description for marmotte_metal_last_error and returns status.
int failWith(int status, const std::string& message);

// Returns the compute pipeline of a kernel function, creating it on first use.
int findPipeline(MetalContext* context, NSString* functionName, id<MTLComputePipelineState>* result);

// Commits without waiting. Completed work is released and its failures recorded for a later wait.
void commitCommandBuffer(MetalContext* context, id<MTLCommandBuffer> commandBuffer);
