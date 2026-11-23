#include "../MetalContext.h"

// count is an element count, not bytes. The managed backend validates shapes and ownership.
// Validate lengths and formats from the allocation registry before encoding anything: all
// three tensors must share the gate's encoding, F32 or F16. A successful return means work
// was committed on the context's queue, not completed.

MARMOTTE_EXPORT int marmotte_metal_swi_glu(MetalContext* context, void* gate, void* up, void* destination, int count) {
    const auto gateAllocation = findBuffer(context, gate, count);
    if (!gateAllocation) return MetalInvalidArgument;

    const auto encoding = gateAllocation->encoding;
    const auto upAllocation = findBuffer(context, up, count, encoding);
    const auto destinationAllocation = findBuffer(context, destination, count, encoding);
    if (!upAllocation || !destinationAllocation) return MetalInvalidArgument;

    @autoreleasepool {
        NSString* functionName = encoding == MetalEncoding::F32 ? @"swiGluF32" : @"swiGluF16";
        return submitElementWise(context, functionName, gateAllocation, upAllocation, destinationAllocation, count);
    }
}
