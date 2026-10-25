namespace Marmotte.Cpu

open System
open System.Buffers
open System.Numerics
open System.Runtime.InteropServices
open Marmotte.Core

/// The numerical part of the CPU backend: F32 kernels over host spans, with no tensors, ownership
/// or lifetime. CpuBackend validates tensors, then calls these functions on the arrays it owns;
/// each function still checks the span lengths it relies on. Safe and vectorized with Vector<T>,
/// most kernels process Vector<single>.Count lanes at a time (4 with ARM64 NEON, 8 with AVX2),
/// then the remaining elements one by one. Destination may be exactly an input, but must not
/// partially overlap one: each vector is read before the same positions are written.
module internal CpuKernels =
    let private requireSameLength (first: ReadOnlySpan<single>) (second: ReadOnlySpan<single>) (destination: Span<single>) =
        if first.Length <> destination.Length || second.Length <> destination.Length then
            invalidArg (nameof destination) "Element-wise kernel spans must have the same length."

    /// destination[i] = left[i] + right[i]. Each lane performs the same IEEE single-precision
    /// addition as a scalar loop, so vector lanes and the tail give identical results.
    let add (left: ReadOnlySpan<single>, right: ReadOnlySpan<single>, destination: Span<single>) =
        requireSameLength left right destination

        let mutable firstScalar = 0
        if Vector.IsHardwareAccelerated then
            let leftVectors = MemoryMarshal.Cast<single, Vector<single>>(left)
            let rightVectors = MemoryMarshal.Cast<single, Vector<single>>(right)
            let destinationVectors = MemoryMarshal.Cast<single, Vector<single>>(destination)
            for index = 0 to destinationVectors.Length - 1 do
                destinationVectors[index] <- leftVectors[index] + rightVectors[index]
            firstScalar <- destinationVectors.Length * Vector<single>.Count

        for index = firstScalar to destination.Length - 1 do
            destination[index] <- left[index] + right[index]

    /// destination[i] = silu(gate[i]) × up[i], with silu(x) = x / (1 + e^-x). Vector lanes use
    /// Vector.Exp and the tail uses MathF.Exp; measured on ARM64, both stay within 2.2e-7 relative
    /// error of a double-precision SiLU above x = -87, and they differ by at most 4 units in the
    /// last place.
    let swiGlu (gate: ReadOnlySpan<single>, up: ReadOnlySpan<single>, destination: Span<single>) =
        requireSameLength gate up destination

        let mutable firstScalar = 0
        if Vector.IsHardwareAccelerated then
            let gateVectors = MemoryMarshal.Cast<single, Vector<single>>(gate)
            let upVectors = MemoryMarshal.Cast<single, Vector<single>>(up)
            let destinationVectors = MemoryMarshal.Cast<single, Vector<single>>(destination)
            for index = 0 to destinationVectors.Length - 1 do
                let x = gateVectors[index]
                let silu = x / (Vector<single>.One + Vector.Exp(-x))
                destinationVectors[index] <- silu * upVectors[index]
            firstScalar <- destinationVectors.Length * Vector<single>.Count

        for index = firstScalar to destination.Length - 1 do
            let x = gate[index]
            let silu = x / (1.0f + MathF.Exp(-x))
            destination[index] <- silu * up[index]

    /// Sum of squares in double precision: rows of thousands of values keep F32 accuracy, and
    /// squares beyond the F32 range, such as that of 1e20, do not overflow.
    let private sumOfSquares (row: ReadOnlySpan<single>) =
        let mutable sum = 0.0
        let mutable firstScalar = 0
        if Vector.IsHardwareAccelerated then
            let vectors = MemoryMarshal.Cast<single, Vector<single>>(row)
            let mutable lowSum = Vector<double>.Zero
            let mutable highSum = Vector<double>.Zero
            for index = 0 to vectors.Length - 1 do
                let low = Vector.WidenLower(vectors[index])
                let high = Vector.WidenUpper(vectors[index])
                lowSum <- lowSum + low * low
                highSum <- highSum + high * high
            sum <- Vector.Sum(lowSum + highSum)
            firstScalar <- vectors.Length * Vector<single>.Count

        for index = firstScalar to row.Length - 1 do
            let value = float row[index]
            sum <- sum + value * value
        sum

    /// destination[c] = row[c] × scale × weight[c], multiplied in that order with two F32 roundings.
    let private scaleRow (row: ReadOnlySpan<single>) (scale: single) (weight: ReadOnlySpan<single>) (destination: Span<single>) =
        let mutable firstScalar = 0
        if Vector.IsHardwareAccelerated then
            let rowVectors = MemoryMarshal.Cast<single, Vector<single>>(row)
            let weightVectors = MemoryMarshal.Cast<single, Vector<single>>(weight)
            let destinationVectors = MemoryMarshal.Cast<single, Vector<single>>(destination)
            let scaleVector = Vector<single>(scale)
            for index = 0 to destinationVectors.Length - 1 do
                destinationVectors[index] <- rowVectors[index] * scaleVector * weightVectors[index]
            firstScalar <- destinationVectors.Length * Vector<single>.Count

        for index = firstScalar to destination.Length - 1 do
            destination[index] <- row[index] * scale * weight[index]

    /// Normalizes consecutive rows of weight.Length values: each row's sum of squares is complete
    /// before the row is written, so destination may be exactly input. The scale
    /// 1 / sqrt(mean + epsilon) is computed in double precision and rounded once to F32.
    let rmsNorm (input: ReadOnlySpan<single>, weight: ReadOnlySpan<single>, epsilon: single, destination: Span<single>) =
        let columns = weight.Length
        let holdsWholeRows = if columns = 0 then input.Length = 0 else input.Length % columns = 0
        if input.Length <> destination.Length || not holdsWholeRows then
            invalidArg (nameof destination) "RMS normalization spans must hold whole rows of the weight's length."

        if columns > 0 then
            for row = 0 to input.Length / columns - 1 do
                let inputRow = input.Slice(row * columns, columns)
                let mean = sumOfSquares inputRow / float columns
                let scale = float32 (1.0 / Math.Sqrt(mean + float epsilon))
                scaleRow inputRow scale weight (destination.Slice(row * columns, columns))

    /// Multiplies left[i] by right[i] at every position of two equal-length spans and returns the
    /// sum of those products (their dot product). Products are formed and summed in double
    /// precision, in two vector accumulators, so long rows keep F32 accuracy even when terms cancel.
    let private sumOfProducts (left: ReadOnlySpan<single>) (right: ReadOnlySpan<single>) =
        let mutable sum = 0.0
        let mutable firstScalar = 0
        if Vector.IsHardwareAccelerated then
            let leftVectors = MemoryMarshal.Cast<single, Vector<single>>(left)
            let rightVectors = MemoryMarshal.Cast<single, Vector<single>>(right)
            let mutable lowSum = Vector<double>.Zero
            let mutable highSum = Vector<double>.Zero
            for index = 0 to leftVectors.Length - 1 do
                let leftVector = leftVectors[index]
                let rightVector = rightVectors[index]
                lowSum <- lowSum + Vector.WidenLower(leftVector) * Vector.WidenLower(rightVector)
                highSum <- highSum + Vector.WidenUpper(leftVector) * Vector.WidenUpper(rightVector)
            sum <- Vector.Sum(lowSum + highSum)
            firstScalar <- leftVectors.Length * Vector<single>.Count

        for index = firstScalar to left.Length - 1 do
            sum <- sum + float left[index] * float right[index]
        sum

    /// destination[t, r] = Σ weights[r, c] × input[t, c], for consecutive token rows of Columns input
    /// values and Rows output values. Decodes each weight row once into a pooled F32 buffer, then
    /// sums its products with the input row of every token, so the whole batch shares the decoding
    /// cost; one token is a matrix-vector product, and an empty batch decodes nothing. Destination
    /// must not overlap input, which every weight row reads in full.
    let matMul (weights: WeightMatrix, input: ReadOnlySpan<single>, destination: Span<single>) =
        let columns = weights.Columns
        let rows = weights.Rows
        let tokens = input.Length / columns
        if input.Length % columns <> 0 || int64 destination.Length <> int64 tokens * int64 rows then
            invalidArg (nameof destination) "Matrix product spans must hold whole token rows of the weight matrix."

        if tokens > 0 then
            let buffer = ArrayPool<single>.Shared.Rent(columns)
            try
                for row = 0 to rows - 1 do
                    weights.DecodeRow(row, Span<single>(buffer, 0, columns))
                    let decodedRow = ReadOnlySpan<single>(buffer, 0, columns)
                    for token = 0 to tokens - 1 do
                        let tokenInput = input.Slice(token * columns, columns)
                        destination[token * rows + row] <- float32 (sumOfProducts decodedRow tokenInput)
            finally
                ArrayPool<single>.Shared.Return(buffer)

    /// Rotates every head of consecutive token rows: pair j of the token at positions[t] turns by
    /// positions[t] × frequency j. The cosine and sine of each angle are computed once per token in
    /// double precision, multiplied by the magnitude, rounded once to F32 and shared by every head.
    /// Each pair is read before it is written, so destination may be exactly input.
    let rope (input: ReadOnlySpan<single>, positions: ReadOnlySpan<int>, embedding: RotaryEmbedding, destination: Span<single>) =
        let tokens = positions.Length
        let headDimension = embedding.HeadDimension
        let valuesPerToken = if tokens = 0 then 0 else input.Length / tokens
        let holdsWholeHeads =
            if tokens = 0 then input.Length = 0
            else input.Length % tokens = 0 && valuesPerToken % headDimension = 0
        if input.Length <> destination.Length || not holdsWholeHeads then
            invalidArg (nameof destination) "RoPE spans must hold whole heads for every token."

        let pairs = headDimension / 2
        let frequencies = Array.init pairs (fun pair -> embedding.Frequencies[pair])
        let magnitude = float embedding.Magnitude
        // Pair j starts at stride × j, and its partner is offset values further.
        let stride, offset =
            match embedding.Pairing with
            | RopePairing.AdjacentPairs -> 2, 1
            | RopePairing.SplitHalves -> 1, pairs

        let cosines = ArrayPool<single>.Shared.Rent(pairs)
        let sines = ArrayPool<single>.Shared.Rent(pairs)
        try
            for token = 0 to tokens - 1 do
                let position = float positions[token]
                for pair = 0 to pairs - 1 do
                    let angle = position * frequencies[pair]
                    cosines[pair] <- float32 (Math.Cos(angle) * magnitude)
                    sines[pair] <- float32 (Math.Sin(angle) * magnitude)

                let tokenInput = input.Slice(token * valuesPerToken, valuesPerToken)
                let tokenOutput = destination.Slice(token * valuesPerToken, valuesPerToken)
                for head = 0 to valuesPerToken / headDimension - 1 do
                    let headStart = head * headDimension
                    for pair = 0 to pairs - 1 do
                        let first = headStart + stride * pair
                        let second = first + offset
                        let a = tokenInput[first]
                        let b = tokenInput[second]
                        tokenOutput[first] <- a * cosines[pair] - b * sines[pair]
                        tokenOutput[second] <- a * sines[pair] + b * cosines[pair]
        finally
            ArrayPool<single>.Shared.Return(cosines)
            ArrayPool<single>.Shared.Return(sines)
