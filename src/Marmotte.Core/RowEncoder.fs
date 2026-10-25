namespace Marmotte.Core

open System
open System.Buffers.Binary

/// Encodes F32 values into stored rows, the inverse of RowDecoder for F32, F16, BF16, Q8_0 and
/// Q4_0, one quantization block at a time. Q4_K and Q6_K are not supported: their reference
/// quantizers search for sub-block scales, which Marmotte does not implement. Q8_0 and Q4_0 follow
/// the block layouts, scales and rounding of the reference quantizers cited in docs/gguf.md
/// (checked 2026-10-25).
///
/// Values that an encoding cannot hold are never silently replaced by plausible numbers. F32
/// keeps every value bit for bit. F16 and BF16 round to the nearest value, ties to even: values
/// beyond their range become infinities, and NaN stays NaN. A Q8_0 or Q4_0 block that contains
/// NaN or an infinity, or whose scale exceeds the finite F16 range, is stored with a NaN scale,
/// so that all of its values decode to NaN; the other blocks of the row are unaffected.
[<RequireQualifiedAccess>]
module RowEncoder =
    /// Whether encode can write rows of an encoding.
    let canEncode (encoding: TensorEncoding) =
        match encoding with
        | TensorEncoding.F32
        | TensorEncoding.F16
        | TensorEncoding.BF16
        | TensorEncoding.Q8_0
        | TensorEncoding.Q4_0 -> true
        | TensorEncoding.Q4_K
        | TensorEncoding.Q6_K -> false

    let private unsupported (encoding: TensorEncoding) : exn =
        NotSupportedException($"Encoding rows as {encoding} is not supported; use F32, F16, BF16, Q8_0 or Q4_0.")

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

    /// A NaN scale decodes every value of the block to NaN, whatever its quants.
    let private writeUnrepresentableBlock (block: Span<byte>) =
        writeHalf block Half.NaN
        block.Slice(2).Clear()

    /// The reciprocal used to compute quants. A block whose stored scale is zero (all zeros, or
    /// values so small that the scale underflows F16) gets zero quants and decodes to zero, without
    /// the infinite reciprocal of an F32 subnormal scale. Quants use the F32 scale before F16
    /// rounding, as the reference quantizers do.
    let private inverseOf (scale: single) (storedScale: Half) =
        if storedScale = Half.Zero then 0.0f else 1.0f / scale

    /// 32 values: scale d = max |x| / 127, then quants round(x / d), half away from zero, in
    /// [-127, 127]. The rounding error of each value is at most d / 2 plus the F16 rounding of d.
    let private encodeQ8_0Block (values: ReadOnlySpan<single>) (block: Span<byte>) =
        let mutable maximum = 0.0f
        let mutable finite = true
        for index = 0 to 31 do
            let value = values[index]
            finite <- finite && Single.IsFinite value
            if abs value > maximum then
                maximum <- abs value

        let scale = maximum / 127.0f
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

    /// 32 values: m is the first value of largest magnitude and d = m / -8, so that m becomes
    /// quant 0, which decodes to -8 × d = m exactly. Quants are x / d + 8.5, truncated and clamped
    /// to [0, 15]: the end opposite to m can only reach 7 × d and clips by one step. Byte j holds
    /// quant j in its low nibble and quant j + 16 in its high nibble.
    let private encodeQ4_0Block (values: ReadOnlySpan<single>) (block: Span<byte>) =
        let mutable signedMaximum = 0.0f
        let mutable finite = true
        for index = 0 to 31 do
            let value = values[index]
            finite <- finite && Single.IsFinite value
            if abs value > abs signedMaximum then
                signedMaximum <- value

        let scale = signedMaximum / -8.0f
        let storedScale = Half.op_Explicit scale
        if not finite || not (Half.IsFinite storedScale) then
            writeUnrepresentableBlock block
        else
            writeHalf block storedScale
            let inverseScale = inverseOf scale storedScale
            let quantize (value: single) = min 15 (max 0 (int (value * inverseScale + 8.5f)))
            for index = 0 to 15 do
                let low = quantize values[index]
                let high = quantize values[index + 16]
                block[2 + index] <- byte (low ||| (high <<< 4))

    /// Encodes values, a whole number of blocks of the encoding, into a row of exactly that many
    /// blocks. Rows of several tensor rows or cache positions can be encoded at once, since blocks
    /// never span rows. Raises NotSupportedException for Q4_K and Q6_K, and ArgumentException when
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
        | TensorEncoding.Q8_0 ->
            for block = 0 to blockCount - 1 do
                encodeQ8_0Block (values.Slice(32 * block, 32)) (destination.Slice(34 * block, 34))
        | TensorEncoding.Q4_0 ->
            for block = 0 to blockCount - 1 do
                encodeQ4_0Block (values.Slice(32 * block, 32)) (destination.Slice(18 * block, 18))
        | TensorEncoding.Q4_K
        | TensorEncoding.Q6_K -> raise (unsupported encoding)
