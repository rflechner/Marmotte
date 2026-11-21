namespace Marmotte.Core

open System
open System.Buffers.Binary

/// Encodes F32 values into stored rows, the inverse of RowDecoder for every encoding except the
/// K-quants Q2_K to Q6_K, one quantization block at a time. Their reference quantizers search for
/// sub-block scales and then quantize those scales, which Marmotte does not implement. Q4_0, Q4_1, Q5_0, Q5_1 and Q8_0 follow the block
/// layouts, scales and rounding of the reference quantizers cited in docs/gguf.md (checked
/// 2026-10-25). IQ4_NL searches its scale in the same way as its reference quantizer without an
/// importance matrix, but is not guaranteed to produce identical blocks.
///
/// Values that an encoding cannot hold are never silently replaced by plausible numbers. F32
/// keeps every value bit for bit. F16 and BF16 round to the nearest value, ties to even: values
/// beyond their range become infinities, and NaN stays NaN. A quantized block that contains NaN or
/// an infinity, or whose scale or minimum exceeds the finite F16 range, is stored with a NaN
/// scale, so that all of its values decode to NaN; the other blocks of the row are unaffected.
[<RequireQualifiedAccess>]
module RowEncoder =
    /// Whether encode can write rows of an encoding.
    let canEncode (encoding: TensorEncoding) =
        match encoding with
        | TensorEncoding.F32
        | TensorEncoding.F16
        | TensorEncoding.BF16
        | TensorEncoding.Q4_0
        | TensorEncoding.Q4_1
        | TensorEncoding.Q5_0
        | TensorEncoding.Q5_1
        | TensorEncoding.Q8_0
        | TensorEncoding.IQ4_NL -> true
        | TensorEncoding.Q2_K
        | TensorEncoding.Q3_K
        | TensorEncoding.Q4_K
        | TensorEncoding.Q5_K
        | TensorEncoding.Q6_K -> false

    let private unsupported (encoding: TensorEncoding) : exn =
        NotSupportedException($"Encoding rows as {encoding} is not supported: K-quant encoders are not implemented.")

    let private writeHalf (destination: Span<byte>) (value: Half) =
        BinaryPrimitives.WriteUInt16LittleEndian(destination, BitConverter.HalfToUInt16Bits value)

    /// The upper 16 bits of the F32 value, rounded to nearest with ties to even by adding a bias
    /// below them. A NaN keeps its sign and upper payload bits, with the quiet bit set, because
    /// truncating a NaN whose payload lies in the lower bits would otherwise give an infinity.
    let private bFloat16Bits (value: single) =
        let bits = BitConverter.SingleToUInt32Bits value
        if Single.IsNaN value then
            uint16 (bits >>> 16) ||| 0x0040us
        else
            let roundingBias = 0x7FFFu + ((bits >>> 16) &&& 1u)
            uint16 ((bits + roundingBias) >>> 16)

    /// A NaN scale decodes every value of the block to NaN, whatever its minimum and quants.
    let private writeUnrepresentableBlock (block: Span<byte>) =
        writeHalf block Half.NaN
        block.Slice(2).Clear()

    /// The reciprocal used to compute quants. A block whose stored scale is zero (all zeros, a
    /// constant block with a minimum, or values so small that the scale underflows F16) gets zero
    /// quants, without the infinite reciprocal of an F32 subnormal scale. Quants use the F32 scale
    /// before F16 rounding, as the reference quantizers do.
    let private inverseOf (scale: single) (storedScale: Half) =
        if storedScale = Half.Zero then 0.0f else 1.0f / scale

    /// The first value of largest magnitude of a block, and whether all of its values are finite.
    let private signedExtreme (values: ReadOnlySpan<single>) =
        let mutable extreme = 0.0f
        let mutable finite = true
        for index = 0 to values.Length - 1 do
            let value = values[index]
            finite <- finite && Single.IsFinite value
            if abs value > abs extreme then
                extreme <- value
        struct (extreme, finite)

    /// The smallest and largest values of a block, and whether all of its values are finite. Strict
    /// comparisons keep the first of equal values, so a minimum of +0 stays +0 before a later -0.
    let private valueRange (values: ReadOnlySpan<single>) =
        let mutable minimum = Single.MaxValue
        let mutable maximum = -Single.MaxValue
        let mutable finite = true
        for index = 0 to values.Length - 1 do
            let value = values[index]
            finite <- finite && Single.IsFinite value
            if value < minimum then
                minimum <- value
            if value > maximum then
                maximum <- value
        struct (minimum, maximum, finite)

    /// Packs 32 quants of 4 bits: byte j holds quant j in its low nibble and quant j + 16 in its
    /// high nibble.
    let private writeFourBitQuants (quantize: single -> int) (values: ReadOnlySpan<single>) (destination: Span<byte>) =
        for index = 0 to 15 do
            destination[index] <- byte (quantize values[index] ||| (quantize values[index + 16] <<< 4))

    /// Packs 32 quants of 5 bits: the fifth bit of quant j into bit j of a little-endian 32-bit
    /// word, then their low 4 bits as writeFourBitQuants does.
    let private writeFiveBitQuants (quantize: single -> int) (values: ReadOnlySpan<single>) (destination: Span<byte>) =
        let mutable fifthBits = 0u
        for index = 0 to 15 do
            let low = quantize values[index]
            let high = quantize values[index + 16]
            destination[4 + index] <- byte ((low &&& 0xF) ||| ((high &&& 0xF) <<< 4))
            fifthBits <- fifthBits ||| (uint32 (low >>> 4) <<< index) ||| (uint32 (high >>> 4) <<< (index + 16))
        BinaryPrimitives.WriteUInt32LittleEndian(destination, fifthBits)

    /// Packs 32 quants of quantBits bits, 4 for Q4_0 and Q4_1 or 5 for Q5_0 and Q5_1.
    let private writeQuants (quantBits: int) (quantize: single -> int) (values: ReadOnlySpan<single>) (destination: Span<byte>) =
        match quantBits with
        | 4 -> writeFourBitQuants quantize values destination
        | 5 -> writeFiveBitQuants quantize values destination
        | _ -> invalidArg (nameof quantBits) $"Quants of {quantBits} bits are not packed by any encoding."

    /// 32 values: scale d = max |x| / 127, then quants round(x / d), half away from zero, in
    /// [-127, 127]. The rounding error of each value is at most d / 2 plus the F16 rounding of d.
    let private encodeQ8_0Block (values: ReadOnlySpan<single>) (block: Span<byte>) =
        let struct (extreme, finite) = signedExtreme values
        let scale = abs extreme / 127.0f
        let storedScale = Half.op_Explicit scale
        if not finite || not (Half.IsFinite storedScale) then
            writeUnrepresentableBlock block
        else
            writeHalf block storedScale
            let inverseScale = inverseOf scale storedScale
            for index = 0 to 31 do
                let rounded = MathF.Round(values[index] * inverseScale, MidpointRounding.AwayFromZero)
                let quant = min 127 (max -127 (int rounded))
                block[2 + index] <- byte (sbyte quant)

    /// Symmetric blocks of quantBits-bit quants with an offset of half their levels: 8 for Q4_0
    /// and 16 for Q5_0. m is the first value of largest magnitude and d = m / -offset, so that m
    /// becomes quant 0, which decodes to -offset × d = m exactly. Quants are x / d + offset + 0.5,
    /// truncated and clamped to [0, 2 × offset - 1]: the end opposite to m can only reach
    /// (offset - 1) × d and clips by one step.
    let private encodeOffsetBlock (quantBits: int) (values: ReadOnlySpan<single>) (block: Span<byte>) =
        let offset = 1 <<< (quantBits - 1)
        let struct (extreme, finite) = signedExtreme values
        let scale = extreme / float32 -offset
        let storedScale = Half.op_Explicit scale
        if not finite || not (Half.IsFinite storedScale) then
            writeUnrepresentableBlock block
        else
            writeHalf block storedScale
            let inverseScale = inverseOf scale storedScale
            let rounding = float32 offset + 0.5f
            let quantize (value: single) = min (2 * offset - 1) (max 0 (int (value * inverseScale + rounding)))
            writeQuants quantBits quantize values (block.Slice(2))

    /// Blocks of quantBits-bit quants with a minimum: Q4_1 with 16 levels and Q5_1 with 32. m is
    /// the smallest value and d = (max - m) / (levels - 1), both stored in F16, so that a block of
    /// positive values uses every level. Quants are (x - m) / d + 0.5, truncated and clamped to
    /// [0, levels - 1]. A constant block has d = 0 and decodes to m.
    let private encodeMinimumBlock (quantBits: int) (values: ReadOnlySpan<single>) (block: Span<byte>) =
        let levels = 1 <<< quantBits
        let struct (minimum, maximum, finite) = valueRange values
        let scale = (maximum - minimum) / float32 (levels - 1)
        let storedScale = Half.op_Explicit scale
        let storedMinimum = Half.op_Explicit minimum
        if not finite || not (Half.IsFinite storedScale) || not (Half.IsFinite storedMinimum) then
            writeUnrepresentableBlock block
        else
            writeHalf block storedScale
            writeHalf (block.Slice(2)) storedMinimum
            let inverseScale = inverseOf scale storedScale
            let quantize (value: single) = min (levels - 1) (max 0 (int ((value - minimum) * inverseScale + 0.5f)))
            writeQuants quantBits quantize values (block.Slice(4))

    /// Index of the IQ4_NL level nearest to a value; a value halfway between two levels takes the
    /// larger one.
    let private nearestLevel (value: double) =
        let levels = RowDecoder.iq4NonLinearLevels
        let mutable index = 0
        while index < levels.Length - 1 && value >= (float levels[index] + float levels[index + 1]) / 2.0 do
            index <- index + 1
        index

    /// Candidate inverse scale 0-15 of an IQ4_NL block whose first value of largest magnitude is
    /// m: m mapped to 127, beyond the largest level 113, then to -134 ... -120 around the
    /// smallest level -127, which is the widest end.
    let private iq4CandidateInverse (extreme: single) (candidate: int) =
        let smallestLevel = float RowDecoder.iq4NonLinearLevels[0]
        if candidate = 0 then -smallestLevel / float extreme
        else (float (candidate - 8) + smallestLevel) / float extreme

    /// 32 values: a scale d and the index of a level per value, value ≈ d × level. Each candidate
    /// inverse scale assigns every value its nearest level; the assignment then gets its weighted
    /// least-squares scale Σ w q x / Σ w q², with weights w = x², so that large values, which
    /// dominate dot products, are reproduced best. The assignment with the smallest weighted error,
    /// that is the largest (Σ w q x)² / Σ w q², gives d. Indices are finally chosen against the
    /// F16 scale actually stored. Sums are computed in double precision.
    let private encodeIq4NlBlock (values: ReadOnlySpan<single>) (block: Span<byte>) =
        let levels = RowDecoder.iq4NonLinearLevels
        let struct (extreme, finite) = signedExtreme values
        let mutable bestScale = 0.0
        let mutable bestScore = 0.0
        if finite && extreme <> 0.0f then
            for candidate = 0 to 15 do
                let inverse = iq4CandidateInverse extreme candidate
                let mutable weightedProducts = 0.0
                let mutable weightedSquares = 0.0
                for index = 0 to 31 do
                    let value = float values[index]
                    let level = float levels[nearestLevel (value * inverse)]
                    let weight = value * value
                    weightedProducts <- weightedProducts + weight * level * value
                    weightedSquares <- weightedSquares + weight * level * level
                if weightedSquares > 0.0 then
                    let score = weightedProducts * weightedProducts / weightedSquares
                    if score > bestScore then
                        bestScore <- score
                        bestScale <- weightedProducts / weightedSquares

        let storedScale = Half.op_Explicit (float32 bestScale)
        if not finite || not (Half.IsFinite storedScale) then
            writeUnrepresentableBlock block
        else
            writeHalf block storedScale
            let inverse = if storedScale = Half.Zero then 0.0 else 1.0 / float storedScale
            writeQuants 4 (fun value -> nearestLevel (float value * inverse)) values (block.Slice(2))

    /// Encodes values, a whole number of blocks of the encoding, into a row of exactly that many
    /// blocks. Rows of several tensor rows or cache positions can be encoded at once, since blocks
    /// never span rows. Raises NotSupportedException for the K-quants, and ArgumentException when
    /// the lengths do not match.
    let encode (encoding: TensorEncoding, values: ReadOnlySpan<single>, destination: Span<byte>) =
        if not (canEncode encoding) then
            raise (unsupported encoding)

        let blockBytes = encoding.BlockBytes
        let blockElements = encoding.BlockElements
        let blockCount = values.Length / blockElements
        if values.Length % blockElements <> 0 || int64 destination.Length <> int64 blockCount * int64 blockBytes then
            invalidArg (nameof destination) $"{values.Length} values cannot encode into a {encoding} row of {destination.Length} bytes."

        match encoding with
        | TensorEncoding.F32 ->
            for index = 0 to blockCount - 1 do
                BinaryPrimitives.WriteSingleLittleEndian(destination.Slice(4 * index, 4), values[index])
        | TensorEncoding.F16 ->
            for index = 0 to blockCount - 1 do
                writeHalf (destination.Slice(2 * index, 2)) (Half.op_Explicit values[index])
        | TensorEncoding.BF16 ->
            for index = 0 to blockCount - 1 do
                BinaryPrimitives.WriteUInt16LittleEndian(destination.Slice(2 * index, 2), bFloat16Bits values[index])
        | TensorEncoding.Q4_0 ->
            for block = 0 to blockCount - 1 do
                encodeOffsetBlock 4 (values.Slice(32 * block, 32)) (destination.Slice(18 * block, 18))
        | TensorEncoding.Q4_1 ->
            for block = 0 to blockCount - 1 do
                encodeMinimumBlock 4 (values.Slice(32 * block, 32)) (destination.Slice(20 * block, 20))
        | TensorEncoding.Q5_0 ->
            for block = 0 to blockCount - 1 do
                encodeOffsetBlock 5 (values.Slice(32 * block, 32)) (destination.Slice(22 * block, 22))
        | TensorEncoding.Q5_1 ->
            for block = 0 to blockCount - 1 do
                encodeMinimumBlock 5 (values.Slice(32 * block, 32)) (destination.Slice(24 * block, 24))
        | TensorEncoding.Q8_0 ->
            for block = 0 to blockCount - 1 do
                encodeQ8_0Block (values.Slice(32 * block, 32)) (destination.Slice(34 * block, 34))
        | TensorEncoding.IQ4_NL ->
            for block = 0 to blockCount - 1 do
                encodeIq4NlBlock (values.Slice(32 * block, 32)) (destination.Slice(18 * block, 18))
        | TensorEncoding.Q2_K
        | TensorEncoding.Q3_K
        | TensorEncoding.Q4_K
        | TensorEncoding.Q5_K
        | TensorEncoding.Q6_K -> raise (unsupported encoding)
