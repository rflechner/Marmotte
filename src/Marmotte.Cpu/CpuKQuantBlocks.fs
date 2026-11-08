namespace Marmotte.Cpu

open System
open System.Buffers.Binary
open System.Runtime.InteropServices
open System.Runtime.Intrinsics
open Marmotte.Core

/// Borrowed quant buffer and owned per-group/block coefficients for one worker.
type internal CpuKQuantRow =
    { Columns: int
      Quants: int16 array
      GroupScales: int array
      GroupMinimums: int array
      BlockScales: single array
      BlockMinimums: single array }

/// Integer views of K-quant rows. Groups of 16 share a factor and optional minimum:
/// weight = factor * quant - minimum. Packed layouts are cited in docs/kernels.md.
module internal CpuKQuantBlocks =
    let createRow columns quants =
        { Columns = columns
          Quants = quants
          GroupScales = Array.zeroCreate (columns / 16)
          GroupMinimums = Array.zeroCreate (columns / 16)
          BlockScales = Array.zeroCreate (columns / 256)
          BlockMinimums = Array.zeroCreate (columns / 256) }

    let supports = function
        | TensorEncoding.Q2_K | TensorEncoding.Q3_K | TensorEncoding.Q4_K
        | TensorEncoding.Q5_K | TensorEncoding.Q6_K -> true
        | _ -> false

    let private half (bytes: ReadOnlySpan<byte>) offset =
        float32 (BitConverter.UInt16BitsToHalf(BinaryPrimitives.ReadUInt16LittleEndian(bytes.Slice(offset, 2))))

    let private scaleAndMinimum (bytes: ReadOnlySpan<byte>) group =
        if group < 4 then
            struct (int bytes[4 + group] &&& 63, int bytes[8 + group] &&& 63)
        else
            struct ((int bytes[8 + group] &&& 15) ||| ((int bytes[group] >>> 6) <<< 4),
                    (int bytes[8 + group] >>> 4) ||| ((int bytes[4 + group] >>> 6) <<< 4))

    let private load (packed: ReadOnlySpan<byte>) offset = Vector128.Create<byte>(packed.Slice(offset, 16))

    let private unpackGroup encoding (packed: ReadOnlySpan<byte>) group (destination: Span<int16>) =
        let laneOffset = 16 * (group % 2)
        let mutable values = Vector128<byte>.Zero
        let mutable offsets = Vector128<byte>.Zero
        match encoding with
        | TensorEncoding.Q2_K | TensorEncoding.Q3_K ->
            let offset = if encoding = TensorEncoding.Q2_K then 16 else 32
            values <- Vector128.ShiftRightLogical(load packed (offset + 32 * (group / 8) + laneOffset), 2 * (group % 8 / 2))
                      &&& Vector128.Create(3uy)
            if encoding = TensorEncoding.Q3_K then
                let high = Vector128.ShiftRightLogical(load packed laneOffset, group / 2) &&& Vector128.Create(1uy)
                offsets <- Vector128.ShiftLeft(Vector128.Create(1uy) - high, 2)
        | TensorEncoding.Q4_K | TensorEncoding.Q5_K ->
            let offset = if encoding = TensorEncoding.Q4_K then 16 else 48
            values <- Vector128.ShiftRightLogical(load packed (offset + 32 * (group / 4) + laneOffset), 4 * (group % 4 / 2))
                      &&& Vector128.Create(15uy)
            if encoding = TensorEncoding.Q5_K then
                let high = Vector128.ShiftRightLogical(load packed (16 + laneOffset), group / 2) &&& Vector128.Create(1uy)
                values <- values ||| Vector128.ShiftLeft(high, 4)
        | TensorEncoding.Q6_K ->
            let halfIndex = group / 8
            let pair = group % 8 / 2
            let low = Vector128.ShiftRightLogical(load packed (halfIndex * 64 + pair % 2 * 32 + laneOffset), 4 * (pair / 2))
                      &&& Vector128.Create(15uy)
            let high = Vector128.ShiftRightLogical(load packed (128 + halfIndex * 32 + laneOffset), 2 * pair)
                       &&& Vector128.Create(3uy)
            values <- low ||| Vector128.ShiftLeft(high, 4)
            offsets <- Vector128.Create(32uy)
        | _ -> invalidArg "encoding" "Expected a K-quant encoding."
        let vectors = MemoryMarshal.Cast<int16, Vector128<int16>>(destination)
        vectors[0] <- Vector128.AsInt16(Vector128.WidenLower(values)) - Vector128.AsInt16(Vector128.WidenLower(offsets))
        vectors[1] <- Vector128.AsInt16(Vector128.WidenUpper(values)) - Vector128.AsInt16(Vector128.WidenUpper(offsets))

    /// Unpack integers without reconstructing F32 weights. Each block has sixteen
    /// integer scales/minimums and two F32 super-block scales. Returns false for
    /// non-finite super-block scales so the caller can use decoded-F32 arithmetic.
    let unpack (weights: WeightMatrix) row (destination: CpuKQuantRow) =
        let quants = destination.Quants.AsSpan(0, destination.Columns)
        let scales = destination.GroupScales
        let minimums = destination.GroupMinimums
        let blockScales = destination.BlockScales
        let blockMinimums = destination.BlockMinimums
        let bytes = weights.Row row
        let encoding = weights.Encoding
        let mutable finite = true
        for block = 0 to weights.Columns / 256 - 1 do
            let packed = bytes.Slice(block * encoding.BlockBytes, encoding.BlockBytes)
            let struct (scale, minimum) =
                match encoding with
                | TensorEncoding.Q2_K -> struct (half packed 80, half packed 82)
                | TensorEncoding.Q3_K -> struct (half packed 108, 0.0f)
                | TensorEncoding.Q4_K | TensorEncoding.Q5_K -> struct (half packed 0, half packed 2)
                | TensorEncoding.Q6_K -> struct (half packed 208, 0.0f)
                | _ -> invalidArg "weights" "Expected a K-quant matrix."
            blockScales[block] <- scale
            blockMinimums[block] <- minimum
            finite <- finite && Single.IsFinite scale && Single.IsFinite minimum
            for group = 0 to 15 do
                let struct (factor, offset) =
                    match encoding with
                    | TensorEncoding.Q2_K -> struct (int packed[group] &&& 15, int packed[group] >>> 4)
                    | TensorEncoding.Q3_K ->
                        let low = (int packed[96 + group % 8] >>> (4 * (group / 8))) &&& 15
                        let high = (int packed[104 + group % 4] >>> (2 * (group / 4))) &&& 3
                        struct ((low ||| (high <<< 4)) - 32, 0)
                    | TensorEncoding.Q4_K | TensorEncoding.Q5_K -> scaleAndMinimum packed (group / 2)
                    | TensorEncoding.Q6_K -> struct (int (sbyte packed[192 + group]), 0)
                    | _ -> invalidArg "weights" "Expected a K-quant matrix."
                scales[block * 16 + group] <- factor
                minimums[block * 16 + group] <- offset
                unpackGroup encoding packed group (quants.Slice(block * 256 + group * 16, 16))
        finite

    // Products fit Int16. After widening, even the worst Q6_K block's scaled
    // sum is bounded by 256 * 32 * 127 * 128 = 133,169,152, within Int32.
    let dot (weights: CpuKQuantRow) (input: CpuQ8Activations) token =
        let columns = weights.Columns
        let quants = ReadOnlySpan<int16>(weights.Quants, 0, columns)
        let scales = weights.GroupScales
        let minimums = weights.GroupMinimums
        let blockScales = weights.BlockScales
        let blockMinimums = weights.BlockMinimums
        let inputQuants = ReadOnlySpan<int16>(input.Quants, token * columns, columns)
        let weightsVectors = MemoryMarshal.Cast<int16, Vector128<int16>>(quants)
        let inputVectors = MemoryMarshal.Cast<int16, Vector128<int16>>(inputQuants)
        let mutable sum = 0.0
        for block = 0 to blockScales.Length - 1 do
            let mutable products = 0
            let mutable corrections = 0
            let mutable vectorProducts = Vector128<int>.Zero
            for group = block * 16 to block * 16 + 15 do
                let inputGroup = token * scales.Length + group
                if Vector128.IsHardwareAccelerated then
                    let first = weightsVectors[group * 2] * inputVectors[group * 2]
                    let second = weightsVectors[group * 2 + 1] * inputVectors[group * 2 + 1]
                    let widened = Vector128.WidenLower(first) + Vector128.WidenUpper(first)
                                  + Vector128.WidenLower(second) + Vector128.WidenUpper(second)
                    vectorProducts <- vectorProducts + widened * Vector128.Create(scales[group])
                else
                    let mutable groupSum = 0
                    for index = group * 16 to group * 16 + 15 do
                        groupSum <- groupSum + int quants[index] * int inputQuants[index]
                    products <- products + groupSum * scales[group]
                corrections <- corrections + minimums[group] * input.Sums[inputGroup]
            if Vector128.IsHardwareAccelerated then products <- Vector128.Sum vectorProducts
            let corrected = float blockScales[block] * float products - float blockMinimums[block] * float corrections
            sum <- sum + float input.Scales[token * blockScales.Length + block] * corrected
        float32 sum
