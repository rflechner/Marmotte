#include "../MetalContext.h"

namespace {
// Encodes one dispatch of a compute function taking (left, right, destination) at indices 0 to 2.
int submitAddition(MetalContext* context, NSString* functionName, const MetalAllocation* left,
                   const MetalAllocation* right, const MetalAllocation* destination, int count) {
    id<MTLComputePipelineState> pipeline = nil;
    const auto status = findPipeline(context, functionName, &pipeline);
    if (status != MetalSuccess) return status;

    id<MTLCommandBuffer> commandBuffer = [context->queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (!encoder) return failWith(MetalExecutionFailure, "Metal could not create a compute encoder.");

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:left->buffer offset:0 atIndex:0];
    [encoder setBuffer:right->buffer offset:0 atIndex:1];
    [encoder setBuffer:destination->buffer offset:0 atIndex:2];

    // The last threadgroup may be smaller than the others: exactly count threads run.
    [encoder dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(count), 1, 1)
       threadsPerThreadgroup:MTLSizeMake(pipeline.threadExecutionWidth, 1, 1)];
    [encoder endEncoding];

    commitCommandBuffer(context, commandBuffer);
    return MetalSuccess;
}
}

// count is an element count, not bytes. The managed backend handles empty tensors.
// Validate ownership, lengths and formats from the allocation registry before encoding anything.
// A successful return means work was committed on the context's queue, not completed.

MARMOTTE_EXPORT int marmotte_metal_add(MetalContext* context, void* left, void* right, void* destination, int count) {
    const auto leftAllocation = findBuffer(context, left, count);
    if (!leftAllocation) return MetalInvalidArgument;

    const auto encoding = leftAllocation->encoding;
    const auto rightAllocation = findBuffer(context, right, count, encoding);
    const auto destinationAllocation = findBuffer(context, destination, count, encoding);
    if (!rightAllocation || !destinationAllocation) return MetalInvalidArgument;

    @autoreleasepool {
        NSString* functionName = encoding == MetalEncoding::F32 ? @"addValuesF32" : @"addValuesF16";
        return submitAddition(context, functionName, leftAllocation, rightAllocation, destinationAllocation, count);
    }
}

// Left and destination are F32 and right is F16. Destination may be left, never right.
MARMOTTE_EXPORT int marmotte_metal_add_f32_f16(MetalContext* context, void* left, void* right, void* destination, int count) {
    const auto leftAllocation = findBuffer(context, left, count, MetalEncoding::F32);
    const auto rightAllocation = findBuffer(context, right, count, MetalEncoding::F16);
    const auto destinationAllocation = findBuffer(context, destination, count, MetalEncoding::F32);
    if (!leftAllocation || !rightAllocation || !destinationAllocation) return MetalInvalidArgument;

    @autoreleasepool {
        return submitAddition(context, @"addValuesF32F16", leftAllocation, rightAllocation, destinationAllocation, count);
    }
}
