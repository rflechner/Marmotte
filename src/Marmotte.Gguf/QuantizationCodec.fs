namespace Marmotte.Gguf

open System
open System.Buffers.Binary
open System.IO

/// Fixed-size GGML block encoding; buffers are owned and reused by the file quantizer.
module internal QuantizationCodec =
    // Wire layouts and rounding conventions:
    // https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-common.h
    // https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.c
    // See block_q4_0, block_q8_0, quantize_row_q4_0_ref and quantize_row_q8_0_ref.
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

    let encodeBlock format (values: single array) (destination: byte array) offset =
        let mutable absoluteMaximum = 0.0f
        let mutable signedMaximum = 0.0f
        for value in values do
            if not (Single.IsFinite value) then
                raise (InvalidDataException("A weight is NaN or infinity."))
            if abs value > absoluteMaximum then
                absoluteMaximum <- abs value
                signedMaximum <- value

        let scale =
            match format with
            | GgufQuantizationFormat.Q4_0 -> signedMaximum / -8.0f
            | GgufQuantizationFormat.Q8_0 -> absoluteMaximum / 127.0f
        let storedScale = Half.op_Explicit scale
        if not (Half.IsFinite storedScale) then
            raise (InvalidDataException("A quantization scale exceeds the finite F16 range."))

        BinaryPrimitives.WriteUInt16LittleEndian(
            destination.AsSpan(offset, 2), BitConverter.HalfToUInt16Bits storedScale)

        // Tiny blocks whose stored scale underflows to zero decode to zero. Avoid
        // an infinite reciprocal for F32 subnormals, including all-zero blocks.
        let inverseScale = if storedScale = Half.Zero then 0.0f else 1.0f / scale
        match format with
        | GgufQuantizationFormat.Q4_0 ->
            let quantize value = min 15 (max 0 (int (value * inverseScale + 8.5f)))
            for index in 0 .. 15 do
                let low = quantize values[index]
                let high = quantize values[index + 16]
                destination[offset + 2 + index] <- byte (low ||| (high <<< 4))
        | GgufQuantizationFormat.Q8_0 ->
            for index in 0 .. 31 do
                let rounded = MathF.Round(values[index] * inverseScale, MidpointRounding.AwayFromZero)
                let quantized = min 127 (max -127 (int rounded))
                destination[offset + 2 + index] <- byte (quantized &&& 255)
