#include <metal_stdlib>

using namespace metal;

kernel void swiGluF32(
    device const float* gate [[buffer(0)]],
    device const float* up [[buffer(1)]],
    device float* destination [[buffer(2)]],
    uint index [[thread_position_in_grid]])
{
    const float x = gate[index];
    destination[index] = x / (1.0f + exp(-x)) * up[index];
}

kernel void swiGluF16(
    device const half* gate [[buffer(0)]],
    device const half* up [[buffer(1)]],
    device half* destination [[buffer(2)]],
    uint index [[thread_position_in_grid]])
{
    const float x = float(gate[index]);
    destination[index] = half(x / (1.0f + exp(-x)) * float(up[index]));
}
