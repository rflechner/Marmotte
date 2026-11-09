#include <metal_stdlib>

using namespace metal;

// In every function below, each thread writes only its own index, so exact input/output aliasing
// is valid. The host dispatches exactly one thread per element, so index needs no bounds check.

kernel void addValuesF32(
    device const float* left [[buffer(0)]],
    device const float* right [[buffer(1)]],
    device float* destination [[buffer(2)]],
    uint index [[thread_position_in_grid]])
{
    destination[index] = left[index] + right[index];
}

// F16 loads and stores use two bytes per value, with no F32 buffer.
kernel void addValuesF16(
    device const half* left [[buffer(0)]],
    device const half* right [[buffer(1)]],
    device half* destination [[buffer(2)]],
    uint index [[thread_position_in_grid]])
{
    destination[index] = left[index] + right[index];
}

// Widening F16 to F32 is exact, so the only rounding is that of the F32 sum.
kernel void addValuesF32F16(
    device const float* left [[buffer(0)]],
    device const half* right [[buffer(1)]],
    device float* destination [[buffer(2)]],
    uint index [[thread_position_in_grid]])
{
    destination[index] = left[index] + float(right[index]);
}
