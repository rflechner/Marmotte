namespace Marmotte.Gguf

open System
open System.Buffers.Binary
open System.IO
open Marmotte.Core

/// Fixed-size GGML block encoding; buffers are owned and reused by the file quantizer.
module internal QuantizationCodec =
    // Wire layouts and rounding conventions are those of Core's RowEncoder; docs/gguf.md cites
    // their reference.
    let inputElementSize = function
        | GgmlType.F32 -> Some 4
        | GgmlType.F16 | GgmlType.BF16 -> Some 2
        | _ -> None

    let layout = function
        | GgufQuantizationFormat.Q4_0 -> GgmlType.Q4_0, GgufFileType.Q4_0, 18
        | GgufQuantizationFormat.Q8_0 -> GgmlType.Q8_0, GgufFileType.Q8_0, 34

    let readValue encoding (source: byte array) offset =
        match encoding with
        | GgmlType.F32 ->
            BinaryPrimitives.ReadSingleLittleEndian(source.AsSpan(offset, 4))
        | GgmlType.F16 ->
            let bits = BinaryPrimitives.ReadUInt16LittleEndian(source.AsSpan(offset, 2))
            float32 (BitConverter.UInt16BitsToHalf bits)
        | GgmlType.BF16 ->
            let bits = BinaryPrimitives.ReadUInt16LittleEndian(source.AsSpan(offset, 2))
            BitConverter.UInt32BitsToSingle(uint32 bits <<< 16)
        | _ -> invalidArg (nameof encoding) "Only F32, F16 and BF16 weights can be decoded."

    /// Encodes one block of 32 weights with Core's RowEncoder, which owns the block arithmetic.
    /// The file quantizer is stricter than RowEncoder: instead of storing a NaN scale, it rejects
    /// non-finite weights and scales beyond the finite F16 range.
    let encodeBlock format (values: single array) (destination: byte array) offset =
        if values |> Array.exists (fun value -> not (Single.IsFinite value)) then
            raise (InvalidDataException("A weight is NaN or infinity."))

        let encoding =
            match format with
            | GgufQuantizationFormat.Q4_0 -> TensorEncoding.Q4_0
            | GgufQuantizationFormat.Q8_0 -> TensorEncoding.Q8_0
        let block = destination.AsSpan(offset, encoding.BlockBytes)
        RowEncoder.encode (encoding, ReadOnlySpan<single>(values), block)

        let storedScale = BitConverter.UInt16BitsToHalf(BinaryPrimitives.ReadUInt16LittleEndian(block))
        if not (Half.IsFinite storedScale) then
            raise (InvalidDataException("A quantization scale exceeds the finite F16 range."))
