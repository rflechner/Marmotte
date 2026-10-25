namespace Marmotte.Cpu

open System
open System.Buffers
open System.Collections.Generic
open System.Numerics
open System.Runtime.InteropServices
open Marmotte.Core

/// F32 kernels over host spans, safe and vectorized with Vector<T>. Each processes
/// Vector<single>.Count lanes at a time (4 with ARM64 NEON, 8 with AVX2), then the remaining
/// elements one by one. Destination may be exactly an input, but must not partially overlap one:
/// each vector is read before the same positions are written.
module private CpuKernels =
    let private requireSameLength (first: ReadOnlySpan<single>) (second: ReadOnlySpan<single>) (destination: Span<single>) =
        if first.Length <> destination.Length || second.Length <> destination.Length then
            invalidArg (nameof destination) "Element-wise kernel spans must have the same length."

    /// destination[i] = left[i] + right[i]. Each lane performs the same IEEE single-precision
    /// addition as a scalar loop, so vector lanes and the tail give identical results.
    let add (left: ReadOnlySpan<single>, right: ReadOnlySpan<single>, destination: Span<single>) =
        requireSameLength left right destination

        let mutable firstScalar = 0
        if Vector.IsHardwareAccelerated then
            let leftVectors = MemoryMarshal.Cast<single, Vector<single>>(left)
            let rightVectors = MemoryMarshal.Cast<single, Vector<single>>(right)
            let destinationVectors = MemoryMarshal.Cast<single, Vector<single>>(destination)
            for index = 0 to destinationVectors.Length - 1 do
                destinationVectors[index] <- leftVectors[index] + rightVectors[index]
            firstScalar <- destinationVectors.Length * Vector<single>.Count

        for index = firstScalar to destination.Length - 1 do
            destination[index] <- left[index] + right[index]

    /// destination[i] = silu(gate[i]) × up[i], with silu(x) = x / (1 + e^-x). Vector lanes use
    /// Vector.Exp and the tail uses MathF.Exp; measured on ARM64, both stay within 2.2e-7 relative
    /// error of a double-precision SiLU above x = -87, and they differ by at most 4 units in the
    /// last place.
    let swiGlu (gate: ReadOnlySpan<single>, up: ReadOnlySpan<single>, destination: Span<single>) =
        requireSameLength gate up destination

        let mutable firstScalar = 0
        if Vector.IsHardwareAccelerated then
            let gateVectors = MemoryMarshal.Cast<single, Vector<single>>(gate)
            let upVectors = MemoryMarshal.Cast<single, Vector<single>>(up)
            let destinationVectors = MemoryMarshal.Cast<single, Vector<single>>(destination)
            for index = 0 to destinationVectors.Length - 1 do
                let x = gateVectors[index]
                let silu = x / (Vector<single>.One + Vector.Exp(-x))
                destinationVectors[index] <- silu * upVectors[index]
            firstScalar <- destinationVectors.Length * Vector<single>.Count

        for index = firstScalar to destination.Length - 1 do
            let x = gate[index]
            let silu = x / (1.0f + MathF.Exp(-x))
            destination[index] <- silu * up[index]

    /// Sum of squares in double precision: rows of thousands of values keep F32 accuracy, and
    /// squares beyond the F32 range, such as that of 1e20, do not overflow.
    let private sumOfSquares (row: ReadOnlySpan<single>) =
        let mutable sum = 0.0
        let mutable firstScalar = 0
        if Vector.IsHardwareAccelerated then
            let vectors = MemoryMarshal.Cast<single, Vector<single>>(row)
            let mutable lowSum = Vector<double>.Zero
            let mutable highSum = Vector<double>.Zero
            for index = 0 to vectors.Length - 1 do
                let low = Vector.WidenLower(vectors[index])
                let high = Vector.WidenUpper(vectors[index])
                lowSum <- lowSum + low * low
                highSum <- highSum + high * high
            sum <- Vector.Sum(lowSum + highSum)
            firstScalar <- vectors.Length * Vector<single>.Count

        for index = firstScalar to row.Length - 1 do
            let value = float row[index]
            sum <- sum + value * value
        sum

    /// destination[c] = row[c] × scale × weight[c], multiplied in that order with two F32 roundings.
    let private scaleRow (row: ReadOnlySpan<single>) (scale: single) (weight: ReadOnlySpan<single>) (destination: Span<single>) =
        let mutable firstScalar = 0
        if Vector.IsHardwareAccelerated then
            let rowVectors = MemoryMarshal.Cast<single, Vector<single>>(row)
            let weightVectors = MemoryMarshal.Cast<single, Vector<single>>(weight)
            let destinationVectors = MemoryMarshal.Cast<single, Vector<single>>(destination)
            let scaleVector = Vector<single>(scale)
            for index = 0 to destinationVectors.Length - 1 do
                destinationVectors[index] <- rowVectors[index] * scaleVector * weightVectors[index]
            firstScalar <- destinationVectors.Length * Vector<single>.Count

        for index = firstScalar to destination.Length - 1 do
            destination[index] <- row[index] * scale * weight[index]

    /// Normalizes consecutive rows of weight.Length values: each row's sum of squares is complete
    /// before the row is written, so destination may be exactly input. The scale
    /// 1 / sqrt(mean + epsilon) is computed in double precision and rounded once to F32.
    let rmsNorm (input: ReadOnlySpan<single>, weight: ReadOnlySpan<single>, epsilon: single, destination: Span<single>) =
        let columns = weight.Length
        let holdsWholeRows = if columns = 0 then input.Length = 0 else input.Length % columns = 0
        if input.Length <> destination.Length || not holdsWholeRows then
            invalidArg (nameof destination) "RMS normalization spans must hold whole rows of the weight's length."

        if columns > 0 then
            for row = 0 to input.Length / columns - 1 do
                let inputRow = input.Slice(row * columns, columns)
                let mean = sumOfSquares inputRow / float columns
                let scale = float32 (1.0 / Math.Sqrt(mean + float epsilon))
                scaleRow inputRow scale weight (destination.Slice(row * columns, columns))

    /// Multiplies left[i] by right[i] at every position of two equal-length spans and returns the
    /// sum of those products (their dot product). Products are formed and summed in double
    /// precision, in two vector accumulators, so long rows keep F32 accuracy even when terms cancel.
    let private sumOfProducts (left: ReadOnlySpan<single>) (right: ReadOnlySpan<single>) =
        let mutable sum = 0.0
        let mutable firstScalar = 0
        if Vector.IsHardwareAccelerated then
            let leftVectors = MemoryMarshal.Cast<single, Vector<single>>(left)
            let rightVectors = MemoryMarshal.Cast<single, Vector<single>>(right)
            let mutable lowSum = Vector<double>.Zero
            let mutable highSum = Vector<double>.Zero
            for index = 0 to leftVectors.Length - 1 do
                let leftVector = leftVectors[index]
                let rightVector = rightVectors[index]
                lowSum <- lowSum + Vector.WidenLower(leftVector) * Vector.WidenLower(rightVector)
                highSum <- highSum + Vector.WidenUpper(leftVector) * Vector.WidenUpper(rightVector)
            sum <- Vector.Sum(lowSum + highSum)
            firstScalar <- leftVectors.Length * Vector<single>.Count

        for index = firstScalar to left.Length - 1 do
            sum <- sum + float left[index] * float right[index]
        sum

    /// destination[r] = Σ weights[r, c] × input[c]. Decodes one weight row at a time into a pooled
    /// F32 buffer of Columns values, then sums its products with input. Destination must not
    /// overlap input, which every row reads in full.
    let matVec (weights: WeightMatrix, input: ReadOnlySpan<single>, destination: Span<single>) =
        if input.Length <> weights.Columns || destination.Length <> weights.Rows then
            invalidArg (nameof destination) "MatVec spans must match the weight matrix columns and rows."

        let columns = weights.Columns
        let buffer = ArrayPool<single>.Shared.Rent(columns)
        try
            for row = 0 to weights.Rows - 1 do
                weights.DecodeRow(row, Span<single>(buffer, 0, columns))
                let decodedRow = ReadOnlySpan<single>(buffer, 0, columns)
                destination[row] <- float32 (sumOfProducts decodedRow input)
        finally
            ArrayPool<single>.Shared.Return(buffer)

/// Storage that its backend releases when the backend itself is disposed.
type private ICpuStorage =
    /// Marks the storage disposed and drops its buffer, without notifying the backend.
    abstract Release: unit -> unit

/// The owned buffer is private to this backend assembly; every public transfer makes a copy.
type private CpuTensor(owner: obj, shape: TensorShape, values: single array, onDispose: ICpuStorage -> unit) =
    let mutable values = values
    let mutable disposed = false

    member _.Owner = owner

    member this.Values =
        ObjectDisposedException.ThrowIf(disposed, this)
        values

    interface ICpuStorage with
        member _.Release() =
            disposed <- true
            values <- Array.empty

    interface ITensor with
        member _.Shape = shape
        member _.Encoding = TensorEncoding.F32
        member _.Backend = BackendKind.Cpu

    interface IDisposable with
        member this.Dispose() =
            if not disposed then
                (this :> ICpuStorage).Release()
                onDispose this

/// Retains the matrix payload without copying it: host memory is already where CPU kernels read.
type private CpuWeightTensor(owner: obj, weights: WeightMatrix, onDispose: ICpuStorage -> unit) =
    let name = weights.Name
    let encoding = weights.Encoding
    let rows = weights.Rows
    let columns = weights.Columns
    let mutable retained = Some weights

    member _.Owner = owner

    /// The stored rows that weight kernels consume without decoding the whole matrix.
    member this.Weights =
        match retained with
        | Some weights -> weights
        | None -> raise (ObjectDisposedException(this.GetType().FullName))

    interface ICpuStorage with
        member _.Release() = retained <- None

    interface IWeightTensor with
        member _.Name = name
        member _.Encoding = encoding
        member _.Rows = rows
        member _.Columns = columns
        member _.Backend = BackendKind.Cpu

    interface IDisposable with
        member this.Dispose() =
            if retained.IsSome then
                (this :> ICpuStorage).Release()
                onDispose this

/// Safe CPU implementation of tensor and embedding contracts, with vectorized Add, SwiGLU and RMS normalization.
/// Owns no native resources. Each instance accepts only tensors that it created, and tracks
/// the live ones so that disposing the backend releases them. Operations complete before
/// returning, which satisfies the ordering contract; Synchronize has nothing to wait for.
[<Sealed>]
type CpuBackend() =
    let owner = obj()
    let liveStorage = HashSet<ICpuStorage>(HashIdentity.Reference)
    let mutable disposed = false

    let forget (storage: ICpuStorage) = liveStorage.Remove(storage) |> ignore

    let track (storage: ICpuStorage) = liveStorage.Add(storage) |> ignore

    let valuesOf argumentName (tensor: ITensor) =
        ArgumentNullException.ThrowIfNull(tensor, argumentName)
        match tensor with
        | :? CpuTensor as tensor when Object.ReferenceEquals(tensor.Owner, owner) -> tensor.Values
        | _ -> invalidArg argumentName "The tensor must belong to this CPU backend instance."

    let weightsOf argumentName (tensor: IWeightTensor) =
        ArgumentNullException.ThrowIfNull(tensor, argumentName)
        match tensor with
        | :? CpuWeightTensor as tensor when Object.ReferenceEquals(tensor.Owner, owner) -> tensor.Weights
        | _ -> invalidArg argumentName "The weights must belong to this CPU backend instance."

    /// Element-wise kernels reject broadcasting: every tensor must have the reference's shape.
    let requireSameShape (operation: string) (reference: ITensor) argumentName (tensor: ITensor) =
        if reference.Shape <> tensor.Shape then
            invalidArg argumentName $"{operation} requires identical shapes, not {reference.Shape} and {tensor.Shape}; broadcasting is not supported."

    /// Requires a vector of the given length, such as a weight per column or a matrix product operand.
    let requireVector (operation: string) argumentName (tensor: ITensor) (length: int) =
        if tensor.Shape.Rank <> 1 || tensor.Shape.Dimensions[0] <> length then
            invalidArg argumentName $"{operation} requires {argumentName} of shape [{length}], not {tensor.Shape}."

    let createTensor shape values =
        let tensor = new CpuTensor(owner, shape, values, forget)
        track tensor
        tensor :> ITensor

    interface ITensorBackend with
        member _.Backend = BackendKind.Cpu

        member this.Allocate(shape) =
            ObjectDisposedException.ThrowIf(disposed, this)
            ArgumentNullException.ThrowIfNull(shape)
            createTensor shape (Array.zeroCreate shape.ElementCount)

        member this.Upload(shape, values) =
            ObjectDisposedException.ThrowIf(disposed, this)
            ArgumentNullException.ThrowIfNull(shape)
            if values.Length <> shape.ElementCount then
                invalidArg (nameof values) "Host value count must match the tensor shape."

            createTensor shape (values.ToArray())

        member this.UploadWeights(weights) =
            ObjectDisposedException.ThrowIf(disposed, this)
            ArgumentNullException.ThrowIfNull(weights)
            let tensor = new CpuWeightTensor(owner, weights, forget)
            track tensor
            tensor :> IWeightTensor

        member this.Download(tensor, destination) =
            ObjectDisposedException.ThrowIf(disposed, this)
            let values = valuesOf (nameof tensor) tensor
            if destination.Length <> values.Length then
                invalidArg (nameof destination) "Host buffer length must match the tensor element count."

            values.AsMemory().CopyTo(destination)

        member this.Synchronize() =
            ObjectDisposedException.ThrowIf(disposed, this)

    interface IAddKernel with
        member this.Add(left, right, destination) =
            ObjectDisposedException.ThrowIf(disposed, this)
            let leftValues = valuesOf (nameof left) left
            let rightValues = valuesOf (nameof right) right
            let destinationValues = valuesOf (nameof destination) destination

            requireSameShape "Add" left (nameof right) right
            requireSameShape "Add" left (nameof destination) destination

            CpuKernels.add (ReadOnlySpan<single>(leftValues), ReadOnlySpan<single>(rightValues), Span<single>(destinationValues))

    interface ISwiGluKernel with
        member this.SwiGlu(gate, up, destination) =
            ObjectDisposedException.ThrowIf(disposed, this)
            let gateValues = valuesOf (nameof gate) gate
            let upValues = valuesOf (nameof up) up
            let destinationValues = valuesOf (nameof destination) destination

            requireSameShape "SwiGLU" gate (nameof up) up
            requireSameShape "SwiGLU" gate (nameof destination) destination

            CpuKernels.swiGlu (ReadOnlySpan<single>(gateValues), ReadOnlySpan<single>(upValues), Span<single>(destinationValues))

    interface IRmsNormKernel with
        member this.RmsNorm(input, weight, epsilon, destination) =
            ObjectDisposedException.ThrowIf(disposed, this)
            let inputValues = valuesOf (nameof input) input
            let weightValues = valuesOf (nameof weight) weight
            let destinationValues = valuesOf (nameof destination) destination

            let shape = input.Shape
            if shape.Rank = 0 then
                invalidArg (nameof input) "RMS normalization requires a rank of at least 1; a scalar has no row to normalize."

            requireVector $"RMS normalization of {shape}" (nameof weight) weight shape.Dimensions[shape.Rank - 1]
            requireSameShape "RMS normalization" input (nameof destination) destination
            if not (Single.IsFinite epsilon) || epsilon <= 0.0f then
                raise (ArgumentOutOfRangeException(nameof epsilon, box epsilon, "Epsilon must be finite and positive."))

            CpuKernels.rmsNorm (ReadOnlySpan<single>(inputValues), ReadOnlySpan<single>(weightValues), epsilon, Span<single>(destinationValues))

    interface IEmbeddingKernel with
        member this.EmbeddingLookup(weights, tokenIds, destination) =
            ObjectDisposedException.ThrowIf(disposed, this)
            let matrix = weightsOf (nameof weights) weights
            let destinationValues = valuesOf (nameof destination) destination
            let dimensions = destination.Shape.Dimensions
            if dimensions.Count <> 2 || dimensions[0] <> tokenIds.Length || dimensions[1] <> matrix.Columns then
                invalidArg (nameof destination) $"Embedding lookup requires shape [{tokenIds.Length}; {matrix.Columns}], not {destination.Shape}."

            let ids = tokenIds.Span
            // Validate the entire batch first: a bad final token must not leave earlier rows written.
            for index = 0 to ids.Length - 1 do
                if ids[index] < 0 || ids[index] >= matrix.Rows then
                    raise (ArgumentOutOfRangeException(nameof tokenIds, $"Token ID {ids[index]} at index {index} must be in [0, {matrix.Rows})."))

            for index = 0 to ids.Length - 1 do
                let output = destinationValues.AsSpan(index * matrix.Columns, matrix.Columns)
                matrix.DecodeRow(ids[index], output)

    interface IMatVecKernel with
        member this.MatVec(weights, input, destination) =
            ObjectDisposedException.ThrowIf(disposed, this)
            let matrix = weightsOf (nameof weights) weights
            let inputValues = valuesOf (nameof input) input
            let destinationValues = valuesOf (nameof destination) destination

            let operation = $"MatVec with {matrix.Rows} × {matrix.Columns} weights"
            requireVector operation (nameof input) input matrix.Columns
            requireVector operation (nameof destination) destination matrix.Rows
            if Object.ReferenceEquals(input, destination) then
                invalidArg (nameof destination) "MatVec destination must differ from its input, which every output reads in full."

            CpuKernels.matVec (matrix, ReadOnlySpan<single>(inputValues), Span<single>(destinationValues))

    interface IDisposable with
        member _.Dispose() =
            if not disposed then
                disposed <- true
                for storage in liveStorage do
                    storage.Release()
                liveStorage.Clear()
