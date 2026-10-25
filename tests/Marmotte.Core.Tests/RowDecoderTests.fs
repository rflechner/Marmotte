module Marmotte.Core.RowDecoderTests

open System
open System.Buffers.Binary
open Marmotte.Core
open Xunit

// Fixtures pack chosen integer quants and scales by hand, following the field descriptions of
// llama.cpp's ggml-common.h and the bit positions written by its reference quantizers. The
// expected values come from those integers, not from the decoder under test. Scales are chosen
// to be exact in half precision.

let private halfBytes (value: single) =
    let bytes = Array.zeroCreate<byte> 2
    BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(), BitConverter.HalfToUInt16Bits(Half.op_Explicit value))
    bytes

let private decode encoding (row: byte array) valueCount =
    let values = Array.zeroCreate<single> valueCount
    RowDecoder.decode (encoding, ReadOnlySpan<byte>(row), Span<single>(values))
    values

/// 32 unsigned 4-bit quants: byte j holds quant j in its low nibble and quant j + 16 in its high one.
let private q4_0Block (scale: single) (quants: int array) =
    let packed = Array.init 16 (fun index -> byte (quants[index] ||| (quants[index + 16] <<< 4)))
    Array.append (halfBytes scale) packed

let private q8_0Block (scale: single) (quants: int array) =
    Array.append (halfBytes scale) (quants |> Array.map (fun quant -> byte (sbyte quant)))

/// Bytes 0-3 hold the low 6 bits of scales 0-3 and the high 2 bits of scales 4-7; bytes 4-7 do
/// the same for minimums; bytes 8-11 hold the low 4 bits of scale and minimum 4-7. Each chunk of
/// 64 quants packs quant l in the low nibble and quant l + 32 in the high nibble of byte l.
let private q4KBlock (scale: single) (minimumScale: single) (scales: int array) (minimums: int array) (quants: int array) =
    let packScaleByte index =
        if index < 4 then
            (scales[index] &&& 63) ||| ((scales[index + 4] >>> 4) <<< 6)
        elif index < 8 then
            (minimums[index - 4] &&& 63) ||| ((minimums[index] >>> 4) <<< 6)
        else
            (scales[index - 4] &&& 15) ||| ((minimums[index - 4] &&& 15) <<< 4)

    let packQuantByte index =
        let chunk = index / 32
        let lane = index % 32
        quants[64 * chunk + lane] ||| (quants[64 * chunk + 32 + lane] <<< 4)

    Array.concat
        [ halfBytes scale
          halfBytes minimumScale
          Array.init 12 (packScaleByte >> byte)
          Array.init 128 (packQuantByte >> byte) ]

let private q4KValues (scale: single) (minimumScale: single) (scales: int array) (minimums: int array) (quants: int array) =
    Array.init 256 (fun value ->
        let subBlock = value / 32
        let factor = scale * float32 scales[subBlock]
        let offset = minimumScale * float32 minimums[subBlock]
        factor * float32 quants[value] - offset)

/// For each half of 128 quants and each lane l, the reference quantizer stores quants l and l + 64
/// in the nibbles of ql[l], quants l + 32 and l + 96 in those of ql[l + 32], and the high 2 bits
/// of the four quants in qh[l], from the lowest bits up. The 16 int8 scales and d follow.
let private q6KBlock (scale: single) (scales: int array) (quants: int array) =
    let lowBits = Array.zeroCreate<byte> 128
    let highBits = Array.zeroCreate<byte> 64
    for half in 0 .. 1 do
        for lane in 0 .. 31 do
            let quantAt group = quants[128 * half + 32 * group + lane]
            lowBits[64 * half + lane] <- byte ((quantAt 0 &&& 15) ||| ((quantAt 2 &&& 15) <<< 4))
            lowBits[64 * half + 32 + lane] <- byte ((quantAt 1 &&& 15) ||| ((quantAt 3 &&& 15) <<< 4))
            let high = (quantAt 0 >>> 4) ||| ((quantAt 1 >>> 4) <<< 2) ||| ((quantAt 2 >>> 4) <<< 4) ||| ((quantAt 3 >>> 4) <<< 6)
            highBits[32 * half + lane] <- byte high

    Array.concat [ lowBits; highBits; scales |> Array.map (fun value -> byte (sbyte value)); halfBytes scale ]

let private q6KValues (scale: single) (scales: int array) (quants: int array) =
    Array.init 256 (fun value -> scale * float32 scales[value / 16] * float32 (quants[value] - 32))

[<Fact>]
let ``decodes floating-point rows including special values`` () =
    let values = [| 1.0f; -2.5f; 65504.0f; -0.0f; 5.9604645e-8f; Single.PositiveInfinity; Single.NaN |]

    let f32 = values |> Array.collect BitConverter.GetBytes
    Assert.Equal<single>(values, decode TensorEncoding.F32 f32 values.Length)

    let f16 = values |> Array.collect halfBytes
    Assert.Equal<single>(values, decode TensorEncoding.F16 f16 values.Length)

    // BF16 keeps the high 16 bits of an F32, so these values have zero low bits.
    let bfloatValues = [| 1.0f; -3.0f; 0.15625f; 1.2676506e30f; Single.NegativeInfinity; Single.NaN |]
    let bf16 =
        bfloatValues
        |> Array.collect (fun value -> BitConverter.GetBytes(uint16 (BitConverter.SingleToUInt32Bits value >>> 16)))
    Assert.Equal<single>(bfloatValues, decode TensorEncoding.BF16 bf16 bfloatValues.Length)

[<Fact>]
let ``decodes Q4_0 blocks with low nibbles first and an offset of 8`` () =
    let first = Array.init 32 (fun index -> (index * 7 + 3) % 16)
    let second = Array.init 32 (fun index -> 15 - index % 16)
    let row = Array.append (q4_0Block 0.5f first) (q4_0Block -0.25f second)
    let expected =
        Array.append
            (first |> Array.map (fun quant -> float32 (quant - 8) * 0.5f))
            (second |> Array.map (fun quant -> float32 (quant - 8) * -0.25f))

    Assert.Equal<single>(expected, decode TensorEncoding.Q4_0 row 64)

[<Fact>]
let ``decodes Q8_0 blocks as signed quants`` () =
    let first = Array.init 32 (fun index -> index * 8 - 128)
    let second = Array.init 32 (fun index -> 127 - index * 3)
    let row = Array.append (q8_0Block 0.0625f first) (q8_0Block 3.0f second)
    let expected =
        Array.append
            (first |> Array.map (fun quant -> float32 quant * 0.0625f))
            (second |> Array.map (fun quant -> float32 quant * 3.0f))

    Assert.Equal<single>(expected, decode TensorEncoding.Q8_0 row 64)

[<Fact>]
let ``decodes Q4_K blocks with 6-bit scales and minimums of every sub-block`` () =
    // Scales and minimums of 16 or more exercise the high bits stored in bytes 0-7.
    let scales = [| 1; 17; 33; 63; 5; 21; 42; 60 |]
    let minimums = [| 0; 63; 16; 31; 48; 7; 32; 59 |]
    let quants = Array.init 256 (fun index -> (index * 5 + 3) % 16)
    let otherQuants = Array.init 256 (fun index -> 15 - index % 16)
    let row =
        Array.append
            (q4KBlock 0.0625f 0.03125f scales minimums quants)
            (q4KBlock 0.5f 0.25f (Array.rev minimums) (Array.rev scales) otherQuants)
    let expected =
        Array.append
            (q4KValues 0.0625f 0.03125f scales minimums quants)
            (q4KValues 0.5f 0.25f (Array.rev minimums) (Array.rev scales) otherQuants)

    Assert.Equal<single>(expected, decode TensorEncoding.Q4_K row 512)

[<Fact>]
let ``decodes Q6_K blocks with signed scales and an offset of 32`` () =
    let scales = Array.init 16 (fun index -> index * 17 - 128)
    let quants = Array.init 256 (fun index -> (index * 11 + 7) % 64)
    let otherScales = Array.init 16 (fun index -> 127 - index * 13)
    let otherQuants = Array.init 256 (fun index -> 63 - index % 64)
    let row = Array.append (q6KBlock 0.125f scales quants) (q6KBlock -0.0078125f otherScales otherQuants)
    let expected =
        Array.append (q6KValues 0.125f scales quants) (q6KValues -0.0078125f otherScales otherQuants)

    Assert.Equal<single>(expected, decode TensorEncoding.Q6_K row 512)

[<Fact>]
let ``rejects rows of partial blocks and destinations of another length`` () =
    let row = q4KBlock 1.0f 1.0f (Array.zeroCreate 8) (Array.zeroCreate 8) (Array.zeroCreate 256)

    Assert.Throws<ArgumentException>(fun () -> decode TensorEncoding.Q4_K row[..142] 256 |> ignore) |> ignore
    Assert.Throws<ArgumentException>(fun () -> decode TensorEncoding.Q4_K row 255 |> ignore) |> ignore
    Assert.Throws<ArgumentException>(fun () -> decode TensorEncoding.F32 [| 0uy; 0uy; 0uy |] 1 |> ignore) |> ignore
