namespace Marmotte.Cpu

open System
open System.Buffers
open System.Runtime.InteropServices
open System.Runtime.Intrinsics
open System.Threading.Tasks
open Marmotte.Core

/// Matrix products over owned host memory. Workers share read-only weights and inputs,
/// write disjoint output columns and finish before returning. No pointers or pinning are used.
module internal CpuMatrixKernels =
    // Four independent accumulation chains reduce dependency latency. Widen before
    // multiplying so finite F32 products do not overflow in F32 before cancellation.
    let private dot128 (left: ReadOnlySpan<single>) (right: ReadOnlySpan<single>) =
        let leftVectors = MemoryMarshal.Cast<single, Vector128<single>>(left)
        let rightVectors = MemoryMarshal.Cast<single, Vector128<single>>(right)
        let mutable lowEven = Vector128<double>.Zero
        let mutable highEven = Vector128<double>.Zero
        let mutable lowOdd = Vector128<double>.Zero
        let mutable highOdd = Vector128<double>.Zero
        let mutable index = 0
        while index + 1 < leftVectors.Length do
            let leftEven = leftVectors[index]
            let rightEven = rightVectors[index]
            let leftOdd = leftVectors[index + 1]
            let rightOdd = rightVectors[index + 1]
            lowEven <- lowEven + Vector128.WidenLower(leftEven) * Vector128.WidenLower(rightEven)
            highEven <- highEven + Vector128.WidenUpper(leftEven) * Vector128.WidenUpper(rightEven)
            lowOdd <- lowOdd + Vector128.WidenLower(leftOdd) * Vector128.WidenLower(rightOdd)
            highOdd <- highOdd + Vector128.WidenUpper(leftOdd) * Vector128.WidenUpper(rightOdd)
            index <- index + 2
        if index < leftVectors.Length then
            lowEven <- lowEven + Vector128.WidenLower(leftVectors[index]) * Vector128.WidenLower(rightVectors[index])
            highEven <- highEven + Vector128.WidenUpper(leftVectors[index]) * Vector128.WidenUpper(rightVectors[index])
        let mutable sum = Vector128.Sum((lowEven + highEven) + (lowOdd + highOdd))
        for column = leftVectors.Length * Vector128<single>.Count to left.Length - 1 do
            sum <- sum + float left[column] * float right[column]
        sum

    // Four independent accumulation chains reduce dependency latency. Widen before
    // multiplying so finite F32 products do not overflow in F32 before cancellation.
    let private dot256 (left: ReadOnlySpan<single>) (right: ReadOnlySpan<single>) =
        let leftVectors = MemoryMarshal.Cast<single, Vector256<single>>(left)
        let rightVectors = MemoryMarshal.Cast<single, Vector256<single>>(right)
        let mutable lowEven = Vector256<double>.Zero
        let mutable highEven = Vector256<double>.Zero
        let mutable lowOdd = Vector256<double>.Zero
        let mutable highOdd = Vector256<double>.Zero
        let mutable index = 0
        while index + 1 < leftVectors.Length do
            let leftEven = leftVectors[index]
            let rightEven = rightVectors[index]
            let leftOdd = leftVectors[index + 1]
            let rightOdd = rightVectors[index + 1]
            lowEven <- lowEven + Vector256.WidenLower(leftEven) * Vector256.WidenLower(rightEven)
            highEven <- highEven + Vector256.WidenUpper(leftEven) * Vector256.WidenUpper(rightEven)
            lowOdd <- lowOdd + Vector256.WidenLower(leftOdd) * Vector256.WidenLower(rightOdd)
            highOdd <- highOdd + Vector256.WidenUpper(leftOdd) * Vector256.WidenUpper(rightOdd)
            index <- index + 2
        if index < leftVectors.Length then
            lowEven <- lowEven + Vector256.WidenLower(leftVectors[index]) * Vector256.WidenLower(rightVectors[index])
            highEven <- highEven + Vector256.WidenUpper(leftVectors[index]) * Vector256.WidenUpper(rightVectors[index])
        let mutable sum = Vector256.Sum((lowEven + highEven) + (lowOdd + highOdd))
        for column = leftVectors.Length * Vector256<single>.Count to left.Length - 1 do
            sum <- sum + float left[column] * float right[column]
        sum

    let private dot (left: ReadOnlySpan<single>) (right: ReadOnlySpan<single>) =
        if Vector256.IsHardwareAccelerated then dot256 left right
        elif Vector128.IsHardwareAccelerated then dot128 left right
        else
            let mutable sum = 0.0
            for column = 0 to left.Length - 1 do
                sum <- sum + float left[column] * float right[column]
            sum

    let private multiplyRows (weights: WeightMatrix) (input: ReadOnlySpan<single>)
                             (destination: Span<single>) firstRow endRow =
        let columns = weights.Columns
        let rows = weights.Rows
        let tokens = input.Length / columns
        let buffer = ArrayPool<single>.Shared.Rent(columns)
        try
            for row = firstRow to endRow - 1 do
                weights.DecodeRow(row, Span<single>(buffer, 0, columns))
                let decoded = ReadOnlySpan<single>(buffer, 0, columns)
                for token = 0 to tokens - 1 do
                    destination[token * rows + row] <- float32 (dot decoded (input.Slice(token * columns, columns)))
        finally
            ArrayPool<single>.Shared.Return(buffer)

    /// destination[t,r] = sum(weights[r,c] * input[t,c]). Each row is decoded once
    /// per batch, using one pooled F32 row per worker. Small products run serially;
    /// larger ones partition contiguous rows. Input and destination must not overlap.
    let matMul (weights: WeightMatrix, input: ReadOnlyMemory<single>, destination: Memory<single>) =
        let columns = weights.Columns
        let rows = weights.Rows
        let tokens = input.Length / columns
        if input.Length % columns <> 0 || int64 destination.Length <> int64 tokens * int64 rows then
            invalidArg (nameof destination) "Matrix product memory must hold whole token rows of the weight matrix."

        if tokens > 0 then
            // An input is limited to Int32 elements and rows to Int32, so this product
            // fits Int64. Keep enough work per partition to amortize task scheduling.
            let work = int64 input.Length * int64 rows
            let workers = int (min (int64 Environment.ProcessorCount) (min (int64 rows) (work / 65536L)))
            if workers <= 1 then
                multiplyRows weights input.Span destination.Span 0 rows
            else
                Parallel.For(0, workers, ParallelOptions(MaxDegreeOfParallelism = workers), fun worker ->
                    let firstRow = int (int64 rows * int64 worker / int64 workers)
                    let endRow = int (int64 rows * int64 (worker + 1) / int64 workers)
                    multiplyRows weights input.Span destination.Span firstRow endRow)
                |> ignore
