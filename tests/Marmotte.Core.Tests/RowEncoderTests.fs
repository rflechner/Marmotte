module Marmotte.Core.RowEncoderTests

open System
open System.Buffers.Binary
open Marmotte.Core
open Xunit

// Expected bytes follow the F16 and BF16 bit layouts, IEEE rounding to nearest with ties to even,
// and the Q8_0 and Q4_0 scales and quant packing of the reference quantizers cited in
// docs/gguf.md, worked out by hand rather than taken from the encoder under test. Round trips decode with RowDecoder, which is
// tested separately against hand-packed blocks.

let private encode (encoding: TensorEncoding) (values: single array) =
    let bytes = Array.zeroCreate<byte> (values.Length / encoding.BlockElements * encoding.BlockBytes)
    RowEncoder.encode (encoding, ReadOnlySpan<single>(values), Span<byte>(bytes))
    bytes

let private decode (encoding: TensorEncoding) (bytes: byte array) valueCount =
    let values = Array.zeroCreate<single> valueCount
    RowDecoder.decode (encoding, ReadOnlySpan<byte>(bytes), Span<single>(values))
    values

let private uint16At (bytes: byte array) (offset: int) =
    BinaryPrimitives.ReadUInt16LittleEndian(ReadOnlySpan<byte>(bytes, offset, 2))

let private bitsOf (values: single array) = values |> Array.map BitConverter.SingleToUInt32Bits

[<Fact>]
let ``F32 rows keep every value bit for bit`` () =
    let values = [| 1.5f; -0.0f; Single.Epsilon; Single.MaxValue; Single.NegativeInfinity; BitConverter.UInt32BitsToSingle 0x7FC01234u |]
    let bytes = encode TensorEncoding.F32 values

    Assert.Equal(0x3FC00000u, BinaryPrimitives.ReadUInt32LittleEndian(ReadOnlySpan<byte>(bytes, 0, 4)))
    Assert.Equal<uint32>(bitsOf values, bitsOf (decode TensorEncoding.F32 bytes values.Length))

[<Fact>]
let ``F16 rows round to nearest with ties to even and overflow to infinity`` () =
    // F16 keeps 10 fraction bits. 1 + 2^-11 lies halfway between fractions 0 and 1 and rounds to
    // the even 0; 1 + 3 × 2^-11 lies halfway between 1 and 2 and rounds to 2. 65,520 lies halfway
    // between the largest finite value, 65,504, and 65,536, beyond the range.
    let values = [| 1.0f; 1.0f + 0.00048828125f; 1.0f + 3.0f * 0.00048828125f; -2.0f; 65504.0f; 65520.0f; -1e6f; Single.NaN |]
    let bytes = encode TensorEncoding.F16 values
    let bits = Array.init values.Length (fun index -> uint16At bytes (2 * index))

    Assert.Equal<uint16>([| 0x3C00us; 0x3C00us; 0x3C02us; 0xC000us; 0x7BFFus; 0x7C00us; 0xFC00us |], bits[..6])
    Assert.True(Half.IsNaN(BitConverter.UInt16BitsToHalf bits[7]))

[<Fact>]
let ``BF16 rows round to nearest with ties to even and keep NaN quiet`` () =
    // BF16 keeps the upper 16 bits of F32, with 7 fraction bits. 1 + 2^-8 lies halfway between
    // fractions 0 and 1 and rounds to 0; 1 + 3 × 2^-8 lies halfway between 1 and 2 and rounds to 2;
    // 1 + 2^-8 + 2^-20 lies just above halfway and rounds up to 1. The largest F32 value rounds to
    // infinity. A NaN whose payload lies only in the lower 16 bits would truncate to an infinity.
    let values =
        [| 1.0f
           1.0f + 0.00390625f
           1.0f + 3.0f * 0.00390625f
           1.0f + 0.00390625f + 9.5367431640625e-7f
           -2.0f
           Single.MaxValue
           Single.NegativeInfinity
           BitConverter.UInt32BitsToSingle 0x7F800001u |]
    let bytes = encode TensorEncoding.BF16 values
    let bits = Array.init values.Length (fun index -> uint16At bytes (2 * index))

    Assert.Equal<uint16>([| 0x3F80us; 0x3F80us; 0x3F82us; 0x3F81us; 0xC000us; 0x7F80us; 0xFF80us; 0x7FC0us |], bits)

[<Fact>]
let ``Q8_0 blocks store the largest magnitude over 127 and round quants half away from zero`` () =
    // The largest magnitude, 127, gives scale 1 (F16 0x3C00); ±0.5 and ±1.5 round away from zero.
    let values = Array.append [| 127.0f; 0.5f; -0.5f; 1.5f; -1.5f; -127.0f |] (Array.zeroCreate 26)
    let bytes = encode TensorEncoding.Q8_0 values

    Assert.Equal<byte>([| 0x00uy; 0x3Cuy; 127uy; 1uy; 255uy; 2uy; 254uy; 129uy |], bytes[..7])
    Assert.All(bytes[8..], fun value -> Assert.Equal(0uy, value))

[<Fact>]
let ``Q4_0 blocks take their scale from the first largest magnitude and pack quants j and j + 16`` () =
    // The first value of largest magnitude, -8, gives scale -8 / -8 = 1 (F16 0x3C00), so value x
    // becomes quant x + 8: values j - 8 give quants j in the low nibbles, and values 7 - j give
    // quants 15 - j in the high nibbles. Every value decodes exactly.
    let values = Array.init 32 (fun index -> if index < 16 then float32 (index - 8) else float32 (23 - index))
    let bytes = encode TensorEncoding.Q4_0 values
    let quants = Array.init 16 (fun index -> byte (index ||| ((15 - index) <<< 4)))

    Assert.Equal<byte>(Array.append [| 0x00uy; 0x3Cuy |] quants, bytes)
    Assert.Equal<single>(values, decode TensorEncoding.Q4_0 bytes 32)

    // A positive largest value gives a negative scale, -1 (F16 0xBC00), and quant 0.
    let positive = encode TensorEncoding.Q4_0 (Array.create 32 8.0f)
    Assert.Equal<byte>(Array.append [| 0x00uy; 0xBCuy |] (Array.zeroCreate 16), positive)

[<Fact>]
let ``encoded rows decode within each encoding's rounding error`` () =
    let random = Random(4243)
    let values = Array.init (32 * 64) (fun _ -> float32 (random.NextDouble() * 20.0 - 10.0))

    // F16 and BF16 keep 11 and 8 significant bits: relative errors of at most 2^-11 and 2^-8.
    for encoding, relativeError in [ TensorEncoding.F16, 0.00048828125f; TensorEncoding.BF16, 0.00390625f ] do
        let decoded = decode encoding (encode encoding values) values.Length
        for index = 0 to values.Length - 1 do
            Assert.InRange(abs (decoded[index] - values[index]), 0.0f, abs values[index] * relativeError + 1e-7f)

    // Q8_0 rounds to half a step of max / 127; Q4_0 can clip the end opposite to its scale by a
    // whole step of max / 8. Rounding the scale to F16 adds at most max × 2^-11.
    for encoding, stepError in [ TensorEncoding.Q8_0, 1.0f / 254.0f; TensorEncoding.Q4_0, 1.0f / 8.0f ] do
        let decoded = decode encoding (encode encoding values) values.Length
        for block = 0 to 63 do
            let blockValues = values[32 * block .. 32 * block + 31]
            let maximum = blockValues |> Array.map abs |> Array.max
            let tolerance = maximum * (stepError + 0.00048828125f) + 1e-6f
            for index = 32 * block to 32 * block + 31 do
                Assert.InRange(abs (decoded[index] - values[index]), 0.0f, tolerance)

[<Fact>]
let ``quantized blocks that cannot be represented decode to NaN without affecting other blocks`` () =
    // 1e7 needs a Q8_0 scale of 78,740 and a Q4_0 scale of -1.25e6, beyond F16's 65,504.
    for encoding in [ TensorEncoding.Q8_0; TensorEncoding.Q4_0 ] do
        for invalid in [ Single.NaN; Single.PositiveInfinity; Single.NegativeInfinity; 1e7f ] do
            let values = Array.init 64 (fun index -> float32 (index % 7) - 3.0f)
            values[40] <- invalid
            let bytes = encode encoding values
            let decoded = decode encoding bytes 64

            Assert.True(Half.IsNaN(BitConverter.UInt16BitsToHalf(uint16At bytes encoding.BlockBytes)))
            Assert.All(decoded[32..], fun value -> Assert.True(Single.IsNaN value))
            Assert.Equal<byte>(encode encoding values[..31], bytes[.. encoding.BlockBytes - 1])

[<Fact>]
let ``zero and subnormal quantized blocks decode to zero`` () =
    for encoding in [ TensorEncoding.Q8_0; TensorEncoding.Q4_0 ] do
        for value in [ 0.0f; Single.Epsilon; -1e-30f ] do
            let decoded = decode encoding (encode encoding (Array.create 32 value)) 32
            Assert.All(decoded, fun actual -> Assert.Equal(0.0f, abs actual))

[<Fact>]
let ``encode rejects K-quants and rows of partial blocks`` () =
    for encoding, values, bytes in [ TensorEncoding.Q4_K, 256, 144; TensorEncoding.Q6_K, 256, 210 ] do
        Assert.False(RowEncoder.canEncode encoding)
        Assert.Throws<NotSupportedException>(fun () ->
            RowEncoder.encode (encoding, ReadOnlySpan<single>(Array.zeroCreate values), Span<byte>(Array.zeroCreate bytes)))
        |> ignore

    Assert.Throws<ArgumentException>(fun () ->
        RowEncoder.encode (TensorEncoding.Q8_0, ReadOnlySpan<single>(Array.zeroCreate 31), Span<byte>(Array.zeroCreate 34)))
    |> ignore
    Assert.Throws<ArgumentException>(fun () ->
        RowEncoder.encode (TensorEncoding.F16, ReadOnlySpan<single>(Array.zeroCreate 3), Span<byte>(Array.zeroCreate 4)))
    |> ignore
