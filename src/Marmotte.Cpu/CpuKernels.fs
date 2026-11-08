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

    /// destination[i] = row[i] × factor, with one F32 multiplication per value in vector lanes and
    /// in the tail alike, so a factor of 1 leaves values unchanged.
    let private multiplyRow (row: ReadOnlySpan<single>) (factor: single) (destination: Span<single>) =
        let mutable firstScalar = 0
        if Vector.IsHardwareAccelerated then
            let rowVectors = MemoryMarshal.Cast<single, Vector<single>>(row)
            let destinationVectors = MemoryMarshal.Cast<single, Vector<single>>(destination)
            let factorVector = Vector<single>(factor)
            for index = 0 to destinationVectors.Length - 1 do
                destinationVectors[index] <- rowVectors[index] * factorVector
            firstScalar <- destinationVectors.Length * Vector<single>.Count

        for index = firstScalar to destination.Length - 1 do
            destination[index] <- row[index] * factor

    /// Multiplies each of the consecutive token rows of input by its own factor:
    /// destination[t, i] = input[t, i] × factors[t]. Destination may be exactly input.
    let scaleTokens (input: ReadOnlySpan<single>, factors: ReadOnlySpan<single>, destination: Span<single>) =
        let tokens = factors.Length
        let holdsWholeRows = if tokens = 0 then input.Length = 0 else input.Length % tokens = 0
        if input.Length <> destination.Length || not holdsWholeRows then
            invalidArg (nameof destination) "Token scaling spans must hold one whole row per factor."

        if tokens > 0 then
            let valuesPerToken = input.Length / tokens
            for token = 0 to tokens - 1 do
                let row = input.Slice(token * valuesPerToken, valuesPerToken)
                multiplyRow row factors[token] (destination.Slice(token * valuesPerToken, valuesPerToken))

    /// Reference attention uses O(prefix length + head dimension) scratch, independent of the
    /// number of heads and tokens. Future cache rows are never decoded, even if they contain NaN.
    let attention (layout: KeyValueCacheLayout, queryHeads: int, position: int,
                   keys: ReadOnlySpan<byte>, values: ReadOnlySpan<byte>,
                   queries: ReadOnlySpan<single>, destination: Span<single>) =
        let dimension = layout.HeadDimension
        let rowValues = queryHeads * dimension
        if queries.Length <> destination.Length || queries.Length % rowValues <> 0 then
            invalidArg (nameof destination) "Attention spans must contain the same whole query rows."
        let tokens = queries.Length / rowValues
        let prefixLength = position + tokens
        if keys.Length / layout.KeyRowBytes < prefixLength || values.Length / layout.ValueRowBytes < prefixLength then
            invalidArg (nameof keys) "Attention requires cached keys and values for every query position."

        let scores = Array.zeroCreate<single> prefixLength
        let decoded = Array.zeroCreate<single> dimension
        let keyHeadBytes = layout.KeyRowBytes / layout.KeyValueHeads
        let valueHeadBytes = layout.ValueRowBytes / layout.KeyValueHeads
        let groupSize = queryHeads / layout.KeyValueHeads
        let scale = 1.0f / MathF.Sqrt(single dimension)

        for token = 0 to tokens - 1 do
            let visible = position + token + 1
            for head = 0 to queryHeads - 1 do
                let cacheHead = head / groupSize
                let query = queries.Slice(token * rowValues + head * dimension, dimension)
                let output = destination.Slice(token * rowValues + head * dimension, dimension)
                let mutable maximum = Single.NegativeInfinity
                for cachedPosition = 0 to visible - 1 do
                    let offset = cachedPosition * layout.KeyRowBytes + cacheHead * keyHeadBytes
                    RowDecoder.decode (layout.KeyEncoding, keys.Slice(offset, keyHeadBytes), decoded.AsSpan())
                    let mutable dot = 0.0f
                    for channel = 0 to dimension - 1 do
                        dot <- dot + query[channel] * decoded[channel]
                    let score = dot * scale
                    scores[cachedPosition] <- score
                    maximum <- MathF.Max(maximum, score)

                let mutable denominator = 0.0f
                for cachedPosition = 0 to visible - 1 do
                    let weight = MathF.Exp(scores[cachedPosition] - maximum)
                    scores[cachedPosition] <- weight
                    denominator <- denominator + weight

                output.Clear()
                for cachedPosition = 0 to visible - 1 do
                    let weight = scores[cachedPosition] / denominator
                    let offset = cachedPosition * layout.ValueRowBytes + cacheHead * valueHeadBytes
                    RowDecoder.decode (layout.ValueEncoding, values.Slice(offset, valueHeadBytes), decoded.AsSpan())
                    for channel = 0 to dimension - 1 do
                        output[channel] <- output[channel] + weight * decoded[channel]
