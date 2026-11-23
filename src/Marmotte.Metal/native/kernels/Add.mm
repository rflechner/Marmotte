#include "../MetalContext.h"

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
        return submitElementWise(context, functionName, leftAllocation, rightAllocation, destinationAllocation, count);
    }
}

// Left and destination are F32 and right is F16. Destination may be left, never right.
MARMOTTE_EXPORT int marmotte_metal_add_f32_f16(MetalContext* context, void* left, void* right, void* destination, int count) {
    const auto leftAllocation = findBuffer(context, left, count, MetalEncoding::F32);
    const auto rightAllocation = findBuffer(context, right, count, MetalEncoding::F16);
    const auto destinationAllocation = findBuffer(context, destination, count, MetalEncoding::F32);
    if (!leftAllocation || !rightAllocation || !destinationAllocation) return MetalInvalidArgument;

    @autoreleasepool {
        return submitElementWise(context, @"addValuesF32F16", leftAllocation, rightAllocation, destinationAllocation, count);
    }
}
