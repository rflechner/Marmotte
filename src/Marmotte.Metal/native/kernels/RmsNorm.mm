#include "../MetalContext.h"

#include <climits>
#include <cstdint>

// Both entry points normalize rows × columns values row by row, with one weight per column. The
// managed backend validates shapes, ownership and epsilon. Validate lengths and formats from the
// allocation registry before encoding anything. A successful return means work was committed on
// the context's queue, not completed.

namespace {
bool validRowCounts(int rows, int columns) {
    return rows > 0 && columns > 0 && static_cast<int64_t>(rows) * columns <= INT_MAX;
}

// Encodes one thread per row; the last threadgroup may be smaller than the others.
int submitRmsNorm(MetalContext* context, NSString* functionName, const MetalAllocation* input,
                  const MetalAllocation* weight, const MetalAllocation* destination, int rows, int columns,
                  float epsilon) {
    id<MTLComputePipelineState> pipeline = nil;
    const auto status = findPipeline(context, functionName, &pipeline);
    if (status != MetalSuccess) return status;

    id<MTLCommandBuffer> commandBuffer = [context->queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (!encoder) return failWith(MetalExecutionFailure, "Metal could not create a compute encoder.");

    const auto columnCount = static_cast<uint32_t>(columns);
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:input->buffer offset:0 atIndex:0];
    [encoder setBuffer:weight->buffer offset:0 atIndex:1];
    [encoder setBuffer:destination->buffer offset:0 atIndex:2];
    [encoder setBytes:&columnCount length:sizeof(columnCount) atIndex:3];
    [encoder setBytes:&epsilon length:sizeof(epsilon) atIndex:4];

    [encoder dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(rows), 1, 1)
       threadsPerThreadgroup:MTLSizeMake(pipeline.threadExecutionWidth, 1, 1)];
    [encoder endEncoding];

    commitCommandBuffer(context, commandBuffer);
    return MetalSuccess;
}
}

// All three tensors share the input's encoding, F32 or F16.
MARMOTTE_EXPORT int marmotte_metal_rms_norm(MetalContext* context, void* input, void* weight, void* destination,
                                           int rows, int columns, float epsilon) {
    if (!validRowCounts(rows, columns)) return MetalInvalidArgument;

    const auto count = rows * columns;
    const auto inputAllocation = findBuffer(context, input, count);
    if (!inputAllocation) return MetalInvalidArgument;

    const auto encoding = inputAllocation->encoding;
    const auto weightAllocation = findBuffer(context, weight, columns, encoding);
    const auto destinationAllocation = findBuffer(context, destination, count, encoding);
    if (!weightAllocation || !destinationAllocation) return MetalInvalidArgument;

    @autoreleasepool {
        NSString* functionName = encoding == MetalEncoding::F32 ? @"rmsNormF32" : @"rmsNormF16";
        return submitRmsNorm(context, functionName, inputAllocation, weightAllocation, destinationAllocation,
                             rows, columns, epsilon);
    }
}

// F32 input and weight, F16 destination: the only mixed combination, requested by name.
MARMOTTE_EXPORT int marmotte_metal_rms_norm_f32_to_f16(MetalContext* context, void* input, void* weight,
                                                      void* destination, int rows, int columns, float epsilon) {
    if (!validRowCounts(rows, columns)) return MetalInvalidArgument;

    const auto count = rows * columns;
    const auto inputAllocation = findBuffer(context, input, count, MetalEncoding::F32);
    const auto weightAllocation = findBuffer(context, weight, columns, MetalEncoding::F32);
    const auto destinationAllocation = findBuffer(context, destination, count, MetalEncoding::F16);
    if (!inputAllocation || !weightAllocation || !destinationAllocation) return MetalInvalidArgument;

    @autoreleasepool {
        return submitRmsNorm(context, @"rmsNormF32ToF16", inputAllocation, weightAllocation, destinationAllocation,
                             rows, columns, epsilon);
    }
}
