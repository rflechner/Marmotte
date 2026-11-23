#include <metal_stdlib>

using namespace metal;

template <typename Input>
static float normalizationScale(device const Input* inputRow, uint columns, float epsilon)
{
    float sumOfSquares = 0.0f;
    for (uint column = 0; column < columns; column++) {
        const float value = float(inputRow[column]);
        sumOfSquares += value * value;
    }

    // Arithmetic may flush subnormals, including epsilon and the squares of normal inputs.
    // For small rows and tiny epsilon, recompute with inputs scaled by 2^32 and epsilon by 2^64.
    // The original sum below 1 bounds the scaled sum far below overflow; above this bound,
    // even at INT_MAX columns, an epsilon below 2^-64 is too small to affect the F32 mean.
    const uint epsilonBits = as_type<uint>(epsilon);
    if (epsilonBits < as_type<uint>(0x1p-64f) && sumOfSquares < 1.0f) {
        float scaledSumOfSquares = 0.0f;
        for (uint column = 0; column < columns; column++) {
            const float value = float(inputRow[column]) * 0x1p32f;
            scaledSumOfSquares += value * value;
        }

        // A positive F32 subnormal is its fraction bits times 2^-149. Reconstruct epsilon
        // times 2^64 directly as a normal F32 value, without arithmetic on the subnormal.
        const float scaledEpsilon = epsilonBits < 0x00800000u
            ? float(epsilonBits) * 0x1p-85f
            : epsilon * 0x1p64f;
        return 0x1p32f / sqrt(scaledSumOfSquares / float(columns) + scaledEpsilon);
    }

    return 1.0f / sqrt(sumOfSquares / float(columns) + epsilon);
}

// One thread normalizes one whole row: the simplest mapping, with parallelism across rows only.
// The host dispatches exactly one thread per row, so row needs no bounds check.
// The squares, their sum and the scale are F32 whatever the storage: squares of F16 values above
// 256 would overflow binary16. Each result is computed in F32 and converted once to the
// destination type. A row is read in full before any of it is written, and each later value is
// read before it is written, so destination may be input, or weight when there is a single row.
template <typename Input, typename Output>
static void normalizeRow(
    device const Input* input,
    device const Input* weight,
    device Output* destination,
    uint columns,
    float epsilon,
    uint row)
{
    device const Input* inputRow = input + row * columns;
    device Output* destinationRow = destination + row * columns;

    const float scale = normalizationScale(inputRow, columns, epsilon);

    for (uint column = 0; column < columns; column++) {
        const float normalized = float(inputRow[column]) * scale * float(weight[column]);
        destinationRow[column] = static_cast<Output>(normalized);
    }
}

kernel void rmsNormF32(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* destination [[buffer(2)]],
    constant uint& columns [[buffer(3)]],
    constant float& epsilon [[buffer(4)]],
    uint row [[thread_position_in_grid]])
{
    normalizeRow(input, weight, destination, columns, epsilon, row);
}

kernel void rmsNormF16(
    device const half* input [[buffer(0)]],
    device const half* weight [[buffer(1)]],
    device half* destination [[buffer(2)]],
    constant uint& columns [[buffer(3)]],
    constant float& epsilon [[buffer(4)]],
    uint row [[thread_position_in_grid]])
{
    normalizeRow(input, weight, destination, columns, epsilon, row);
}

// F32 input and weight, F16 destination: the F32 result is rounded to the nearest F16, and values
// beyond the F16 range become infinities. Different storage types mean no aliasing.
kernel void rmsNormF32ToF16(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device half* destination [[buffer(2)]],
    constant uint& columns [[buffer(3)]],
    constant float& epsilon [[buffer(4)]],
    uint row [[thread_position_in_grid]])
{
    normalizeRow(input, weight, destination, columns, epsilon, row);
}
