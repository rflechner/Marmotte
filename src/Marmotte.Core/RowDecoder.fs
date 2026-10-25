namespace Marmotte.Core

open System
open System.Buffers.Binary

/// Decodes stored rows into F32 values, one quantization block at a time, without decoding a
/// whole matrix. Block layouts and arithmetic follow the reference block definitions and
/// reconstruction functions cited in docs/kernels.md (checked 2026-10-25). Each value uses the
/// same F32 operations in the same order as those functions, so the results match unless a C
/// compiler fuses their multiplications with the following addition or subtraction.
[<RequireQualifiedAccess>]
module RowDecoder =
    let private readHalf (bytes: ReadOnlySpan<byte>) =
        float32 (BitConverter.UInt16BitsToHalf(BinaryPrimitives.ReadUInt16LittleEndian(bytes)))

    /// 32 values: a half scale, then 16 bytes whose low nibbles hold values 0-15 and whose high
    /// nibbles hold values 16-31, as unsigned 4-bit quants offset by 8.
    let private decodeQ4_0Block (block: ReadOnlySpan<byte>) (destination: Span<single>) =
        let scale = readHalf block
        let quants = block.Slice(2, 16)
        for index = 0 to 15 do
            let packed = int quants[index]
            destination[index] <- float32 ((packed &&& 0xF) - 8) * scale
            destination[index + 16] <- float32 ((packed >>> 4) - 8) * scale

    /// 32 values: a half scale d, a half minimum m, then 16 bytes of unsigned 4-bit quants packed
    /// as in Q4_0. Value = q × d + m.
    let private decodeQ4_1Block (block: ReadOnlySpan<byte>) (destination: Span<single>) =
        let scale = readHalf block
        let minimum = readHalf (block.Slice(2))
        let quants = block.Slice(4, 16)
        for index = 0 to 15 do
            let packed = int quants[index]
            destination[index] <- float32 (packed &&& 0xF) * scale + minimum
            destination[index + 16] <- float32 (packed >>> 4) * scale + minimum

    /// Unsigned 5-bit quant j of a Q5_0 or Q5_1 block, whose fifth bit is bit j of a little-endian
    /// 32-bit word and whose low 4 bits are packed as in Q4_0.
    let private fiveBitQuant (fifthBits: uint32) (lowBits: ReadOnlySpan<byte>) (index: int) =
        let low = if index < 16 then int lowBits[index] &&& 0xF else int lowBits[index - 16] >>> 4
        low ||| (int ((fifthBits >>> index) &&& 1u) <<< 4)

    /// 32 values: a half scale, the 4 bytes of fifth bits, then 16 bytes of low 4 bits. Quants are
    /// offset by 16, as Q4_0's are by 8.
    let private decodeQ5_0Block (block: ReadOnlySpan<byte>) (destination: Span<single>) =
        let scale = readHalf block
        let fifthBits = BinaryPrimitives.ReadUInt32LittleEndian(block.Slice(2, 4))
        let lowBits = block.Slice(6, 16)
        for index = 0 to 31 do
            destination[index] <- float32 (fiveBitQuant fifthBits lowBits index - 16) * scale

    /// 32 values: a half scale d, a half minimum m, the 4 bytes of fifth bits, then 16 bytes of low
    /// 4 bits. Value = q × d + m.
    let private decodeQ5_1Block (block: ReadOnlySpan<byte>) (destination: Span<single>) =
        let scale = readHalf block
        let minimum = readHalf (block.Slice(2))
        let fifthBits = BinaryPrimitives.ReadUInt32LittleEndian(block.Slice(4, 4))
        let lowBits = block.Slice(8, 16)
        for index = 0 to 31 do
            destination[index] <- float32 (fiveBitQuant fifthBits lowBits index) * scale + minimum

    /// The 16 levels of IQ4_NL in increasing order: closer together near zero, where most values
    /// of a block lie, than the uniform levels of Q4_0. RowEncoder searches the same levels.
    let internal iq4NonLinearLevels =
        [| -127.0f; -104.0f; -83.0f; -65.0f; -49.0f; -35.0f; -22.0f; -10.0f
           1.0f; 13.0f; 25.0f; 38.0f; 53.0f; 69.0f; 89.0f; 113.0f |]

    /// 32 values: a half scale d, then 16 bytes of 4-bit level indices packed as in Q4_0.
    /// Value = d × level.
    let private decodeIq4NlBlock (block: ReadOnlySpan<byte>) (destination: Span<single>) =
        let scale = readHalf block
        let indices = block.Slice(2, 16)
        for index = 0 to 15 do
            let packed = int indices[index]
            destination[index] <- scale * iq4NonLinearLevels[packed &&& 0xF]
            destination[index + 16] <- scale * iq4NonLinearLevels[packed >>> 4]

    /// 32 values: a half scale, then 32 signed 8-bit quants.
    let private decodeQ8_0Block (block: ReadOnlySpan<byte>) (destination: Span<single>) =
        let scale = readHalf block
        let quants = block.Slice(2, 32)
        for index = 0 to 31 do
            destination[index] <- float32 (sbyte quants[index]) * scale

    /// The 6-bit scale and minimum of a Q4_K sub-block (0-7) from its 12 packed bytes. Bytes 0-3
    /// hold the scales of sub-blocks 0-3 and bytes 4-7 their minimums, in their low 6 bits.
    /// Sub-blocks 4-7 keep their low 4 bits in the nibbles of bytes 8-11, and their high 2 bits
    /// in the top bits of bytes 0-3 (scales) and 4-7 (minimums).
    let private q4KScaleAndMinimum (packed: ReadOnlySpan<byte>) (subBlock: int) =
        if subBlock < 4 then
            struct (int packed[subBlock] &&& 63, int packed[subBlock + 4] &&& 63)
        else
            let shared = int packed[subBlock + 4]
            let scale = (shared &&& 0xF) ||| ((int packed[subBlock - 4] >>> 6) <<< 4)
            let minimum = (shared >>> 4) ||| ((int packed[subBlock] >>> 6) <<< 4)
            struct (scale, minimum)

    /// 256 values in eight sub-blocks of 32: value = d × scale × quant - dmin × minimum, with
    /// both products of the half super-block scales computed once per sub-block. Each chunk of
    /// 64 values reads 32 quant bytes: their low nibbles first, then their high nibbles.
    let private decodeQ4KBlock (block: ReadOnlySpan<byte>) (destination: Span<single>) =
        let scale = readHalf block
        let minimumScale = readHalf (block.Slice(2))
        let packedScales = block.Slice(4, 12)
        let quants = block.Slice(16, 128)

        for chunk = 0 to 3 do
            let chunkQuants = quants.Slice(32 * chunk, 32)
            let output = destination.Slice(64 * chunk, 64)
            let struct (lowScale, lowMinimum) = q4KScaleAndMinimum packedScales (2 * chunk)
            let struct (highScale, highMinimum) = q4KScaleAndMinimum packedScales (2 * chunk + 1)
            let lowFactor = scale * float32 lowScale
            let lowOffset = minimumScale * float32 lowMinimum
            let highFactor = scale * float32 highScale
            let highOffset = minimumScale * float32 highMinimum

            for index = 0 to 31 do
                let packed = int chunkQuants[index]
                output[index] <- lowFactor * float32 (packed &&& 0xF) - lowOffset
                output[index + 32] <- highFactor * float32 (packed >>> 4) - highOffset

    /// 256 values: 128 bytes of low 4 bits, 64 bytes of high 2 bits, 16 signed 8-bit scales (one
    /// per 16 values), then the half super-block scale. Value = d × scale × (quant - 32). Each half
    /// of 128 values is four groups of 32: groups 0 and 1 use the low nibbles of 64 bytes, groups
    /// 2 and 3 their high nibbles, and group g takes its high bits from bits 2g and 2g + 1.
    let private decodeQ6KBlock (block: ReadOnlySpan<byte>) (destination: Span<single>) =
        let lowBits = block.Slice(0, 128)
        let highBits = block.Slice(128, 64)
        let scales = block.Slice(192, 16)
        let scale = readHalf (block.Slice(208))

        for half = 0 to 1 do
            let halfLowBits = lowBits.Slice(64 * half, 64)
            let halfHighBits = highBits.Slice(32 * half, 32)
            let halfScales = scales.Slice(8 * half, 8)
            let output = destination.Slice(128 * half, 128)

            for group = 0 to 3 do
                let groupLowBits = halfLowBits.Slice(32 * (group % 2), 32)
                let lowShift = if group < 2 then 0 else 4
                for lane = 0 to 31 do
                    let low = (int groupLowBits[lane] >>> lowShift) &&& 0xF
                    let high = (int halfHighBits[lane] >>> (2 * group)) &&& 3
                    let quant = (low ||| (high <<< 4)) - 32
                    let value = 32 * group + lane
                    output[value] <- scale * float32 (sbyte halfScales[value / 16]) * float32 quant

    /// Decodes one stored row, which must hold whole blocks of the encoding, into BlockElements
    /// values per block. F32, F16 and BF16 rows are read as little-endian values; NaN and
    /// infinities are preserved. Raises ArgumentException when the lengths do not match.
    let decode (encoding: TensorEncoding, row: ReadOnlySpan<byte>, destination: Span<single>) =
        let blockBytes = encoding.BlockBytes
        let blockElements = encoding.BlockElements
        let blockCount = row.Length / blockBytes
        if row.Length % blockBytes <> 0 || int64 destination.Length <> int64 blockCount * int64 blockElements then
            invalidArg (nameof destination) $"A {encoding} row of {row.Length} bytes cannot decode into {destination.Length} values."

        match encoding with
        | TensorEncoding.F32 ->
            for index = 0 to blockCount - 1 do
                destination[index] <- BinaryPrimitives.ReadSingleLittleEndian(row.Slice(4 * index, 4))
        | TensorEncoding.F16 ->
            for index = 0 to blockCount - 1 do
                destination[index] <- readHalf (row.Slice(2 * index, 2))
        | TensorEncoding.BF16 ->
            for index = 0 to blockCount - 1 do
                let bits = uint32 (BinaryPrimitives.ReadUInt16LittleEndian(row.Slice(2 * index, 2)))
                destination[index] <- BitConverter.UInt32BitsToSingle(bits <<< 16)
        | TensorEncoding.Q4_0 ->
            for block = 0 to blockCount - 1 do
                decodeQ4_0Block (row.Slice(18 * block, 18)) (destination.Slice(32 * block, 32))
        | TensorEncoding.Q4_1 ->
            for block = 0 to blockCount - 1 do
                decodeQ4_1Block (row.Slice(20 * block, 20)) (destination.Slice(32 * block, 32))
        | TensorEncoding.Q5_0 ->
            for block = 0 to blockCount - 1 do
                decodeQ5_0Block (row.Slice(22 * block, 22)) (destination.Slice(32 * block, 32))
        | TensorEncoding.Q5_1 ->
            for block = 0 to blockCount - 1 do
                decodeQ5_1Block (row.Slice(24 * block, 24)) (destination.Slice(32 * block, 32))
        | TensorEncoding.Q8_0 ->
            for block = 0 to blockCount - 1 do
                decodeQ8_0Block (row.Slice(34 * block, 34)) (destination.Slice(32 * block, 32))
        | TensorEncoding.IQ4_NL ->
            for block = 0 to blockCount - 1 do
                decodeIq4NlBlock (row.Slice(18 * block, 18)) (destination.Slice(32 * block, 32))
        | TensorEncoding.Q4_K ->
            for block = 0 to blockCount - 1 do
                decodeQ4KBlock (row.Slice(144 * block, 144)) (destination.Slice(256 * block, 256))
        | TensorEncoding.Q6_K ->
            for block = 0 to blockCount - 1 do
                decodeQ6KBlock (row.Slice(210 * block, 210)) (destination.Slice(256 * block, 256))
