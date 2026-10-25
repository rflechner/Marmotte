namespace Marmotte.Cpu

open System
open System.Collections.Generic
open Marmotte.Core

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

/// Safe CPU implementation of the tensor and kernel contracts. This type is the plumbing: storage,
/// ownership, lifetime and argument validation, while CpuKernels does the arithmetic. It owns no
/// native resources. Each instance accepts only tensors that it created, and tracks the live ones
/// so that disposing the backend releases them. Operations complete before returning, which
/// satisfies the ordering contract; Synchronize has nothing to wait for.
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

    /// Position-based kernels take one token per first-dimension row.
    let requireTokenRows (operation: string) argumentName (tensor: ITensor) (tokens: int) =
        let shape = tensor.Shape
        if shape.Rank < 2 || shape.Dimensions[0] <> tokens then
            invalidArg argumentName $"{operation} of {tokens} positions requires {argumentName} of rank 2 or more with {tokens} tokens, not {shape}."

    /// Checks every position before any write, so a bad last position leaves the destination unchanged.
    let requireNonNegativePositions (positions: ReadOnlySpan<int>) =
        for index = 0 to positions.Length - 1 do
            if positions[index] < 0 then
                raise (ArgumentOutOfRangeException(nameof positions, $"Position {positions[index]} at index {index} must be at least 0."))

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

            CpuKernels.matMul (matrix, ReadOnlySpan<single>(inputValues), Span<single>(destinationValues))

    interface IMatMulKernel with
        member this.MatMul(weights, input, destination) =
            ObjectDisposedException.ThrowIf(disposed, this)
            let matrix = weightsOf (nameof weights) weights
            let inputValues = valuesOf (nameof input) input
            let destinationValues = valuesOf (nameof destination) destination

            let operation = $"MatMul with {matrix.Rows} × {matrix.Columns} weights"
            let inputDimensions = input.Shape.Dimensions
            if inputDimensions.Count <> 2 || inputDimensions[1] <> matrix.Columns then
                invalidArg (nameof input) $"{operation} requires input of shape [tokens; {matrix.Columns}], not {input.Shape}."

            let tokens = inputDimensions[0]
            let destinationDimensions = destination.Shape.Dimensions
            if destinationDimensions.Count <> 2 || destinationDimensions[0] <> tokens || destinationDimensions[1] <> matrix.Rows then
                invalidArg (nameof destination) $"{operation} requires destination of shape [{tokens}; {matrix.Rows}], not {destination.Shape}."
            if Object.ReferenceEquals(input, destination) then
                invalidArg (nameof destination) "MatMul destination must differ from its input, which every weight row reads in full."

            CpuKernels.matMul (matrix, ReadOnlySpan<single>(inputValues), Span<single>(destinationValues))

    interface IRopeKernel with
        member this.Rope(input, positions, embedding, destination) =
            ObjectDisposedException.ThrowIf(disposed, this)
            ArgumentNullException.ThrowIfNull(embedding)
            let inputValues = valuesOf (nameof input) input
            let destinationValues = valuesOf (nameof destination) destination

            requireTokenRows "RoPE" (nameof input) input positions.Length
            let shape = input.Shape
            let valuesPerToken = shape.Dimensions |> Seq.skip 1 |> Seq.fold (fun product dimension -> product * int64 dimension) 1L
            if valuesPerToken % int64 embedding.HeadDimension <> 0L then
                invalidArg (nameof input) $"RoPE requires whole heads of {embedding.HeadDimension} values per token, not {shape}."
            requireSameShape "RoPE" input (nameof destination) destination

            let ids = positions.Span
            requireNonNegativePositions ids

            CpuKernels.rope (ReadOnlySpan<single>(inputValues), ids, embedding, Span<single>(destinationValues))

    interface IAttentionTemperatureKernel with
        member this.ApplyAttentionTemperature(queries, positions, temperature, destination) =
            ObjectDisposedException.ThrowIf(disposed, this)
            ArgumentNullException.ThrowIfNull(temperature)
            let queryValues = valuesOf (nameof queries) queries
            let destinationValues = valuesOf (nameof destination) destination

            requireTokenRows "Attention temperature" (nameof queries) queries positions.Length
            requireSameShape "Attention temperature" queries (nameof destination) destination
            let ids = positions.Span
            requireNonNegativePositions ids

            let factors = Array.zeroCreate<single> ids.Length
            for index = 0 to ids.Length - 1 do
                factors[index] <- temperature.Factor(ids[index])

            CpuKernels.scaleTokens (ReadOnlySpan<single>(queryValues), ReadOnlySpan<single>(factors), Span<single>(destinationValues))

    interface IDisposable with
        member _.Dispose() =
            if not disposed then
                disposed <- true
                for storage in liveStorage do
                    storage.Release()
                liveStorage.Clear()
