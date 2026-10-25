namespace Marmotte.Cpu

open System
open System.Buffers.Binary
open Marmotte.Core

/// Safe scalar decoding into caller-owned F32 storage. No temporary row or block buffers.
/// Layouts verified on 2026-10-25 against block_q4_0, block_q8_0, block_q4_K, block_q6_K in
/// https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-common.h
/// and the reconstruction formulas in
/// https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.c.
/// Marmotte addresses each output group explicitly rather than using native packed structs.
module internal WeightRowDecoder =
    let private readHalf (bytes: ReadOnlySpan<byte>) offset =
        let bits = BinaryPrimitives.ReadUInt16LittleEndian(bytes.Slice(offset, 2))
        float32 (BitConverter.UInt16BitsToHalf bits)

    let private signedByte (value: byte) =
        if value < 128uy then int value else int value - 256

    let private decodeQ4Zero (block: ReadOnlySpan<byte>) (destination: Span<single>) =
        let scale = readHalf block 0
        for index = 0 to 31 do
            let packed = int block[2 + index % 16]
            let quantized = (packed >>> (4 * (index / 16))) &&& 15
            destination[index] <- scale * single (quantized - 8)

    let private decodeQ8Zero (block: ReadOnlySpan<byte>) (destination: Span<single>) =
        let scale = readHalf block 0
        for index = 0 to 31 do
            destination[index] <- scale * single (signedByte block[2 + index])

    // The first four scales/minima occupy six low bits each. The remaining four combine
    // nibbles in the final four bytes with the two high bits of those earlier bytes.
    let private q4KFactor (packed: ReadOnlySpan<byte>) group isMinimum =
        let bank = if isMinimum then 4 else 0
        if group < 4 then
            int packed[bank + group] &&& 63
        else
            let shift = if isMinimum then 4 else 0
            let low = (int packed[4 + group] >>> shift) &&& 15
            let high = int packed[bank + group - 4] >>> 6
            low ||| (high <<< 4)

    let private decodeQ4K (block: ReadOnlySpan<byte>) (destination: Span<single>) =
        let scale = readHalf block 0
        let minimumScale = readHalf block 2
        let factors = block.Slice(4, 12)
        for group = 0 to 7 do
            let groupScale = scale * single (q4KFactor factors group false)
            let groupMinimum = minimumScale * single (q4KFactor factors group true)
            let quantOffset = 16 + (group / 2) * 32
            let shift = (group % 2) * 4
            for lane = 0 to 31 do
                let quantized = (int block[quantOffset + lane] >>> shift) &&& 15
                destination[group * 32 + lane] <- groupScale * single quantized - groupMinimum

    let private decodeQ6K (block: ReadOnlySpan<byte>) (destination: Span<single>) =
        let scale = readHalf block 208
        for group = 0 to 15 do
            let groupScale = scale * single (signedByte block[192 + group])
            for lane = 0 to 15 do
                let index = group * 16 + lane
                // Each 128-value half stores four 32-value planes. Low nibbles pair planes
                // 0/2 and 1/3; one high-bit byte contributes two bits to each plane.
                let half = index / 128
                let plane = (index % 128) / 32
                let position = index % 32
                let lowByte = int block[half * 64 + (plane % 2) * 32 + position]
                let low = (lowByte >>> ((plane / 2) * 4)) &&& 15
                let highByte = int block[128 + half * 32 + position]
                let high = (highByte >>> (plane * 2)) &&& 3
                let quantized = (low ||| (high <<< 4)) - 32
                destination[index] <- groupScale * single quantized

    /// WeightMatrix has already validated row sizes, block boundaries and payload length.
    /// The destination must be exactly one row; a bad length is rejected before writing.
    let decode (weights: WeightMatrix) row (destination: Span<single>) =
        if destination.Length <> weights.Columns then
            invalidArg (nameof destination) "Decoded row length must equal the matrix column count."

        let bytes = weights.Row(row)
        match weights.Encoding with
        | TensorEncoding.F32 ->
            for index = 0 to destination.Length - 1 do
                destination[index] <- BinaryPrimitives.ReadSingleLittleEndian(bytes.Slice(index * 4, 4))
        | TensorEncoding.F16 ->
            for index = 0 to destination.Length - 1 do
                destination[index] <- readHalf bytes (index * 2)
        | TensorEncoding.BF16 ->
            for index = 0 to destination.Length - 1 do
                let bits = BinaryPrimitives.ReadUInt16LittleEndian(bytes.Slice(index * 2, 2))
                destination[index] <- BitConverter.UInt32BitsToSingle(uint32 bits <<< 16)
        | TensorEncoding.Q4_0
        | TensorEncoding.Q8_0
        | TensorEncoding.Q4_K
        | TensorEncoding.Q6_K ->
            let blockElements = weights.Encoding.BlockElements
            let blockBytes = weights.Encoding.BlockBytes
            for blockIndex = 0 to weights.Columns / blockElements - 1 do
                let block = bytes.Slice(blockIndex * blockBytes, blockBytes)
                let output = destination.Slice(blockIndex * blockElements, blockElements)
                match weights.Encoding with
                | TensorEncoding.Q4_0 -> decodeQ4Zero block output
                | TensorEncoding.Q8_0 -> decodeQ8Zero block output
                | TensorEncoding.Q4_K -> decodeQ4K block output
                | TensorEncoding.Q6_K -> decodeQ6K block output
                | _ -> invalidOp "Expected a quantized encoding."
