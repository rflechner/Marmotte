namespace Marmotte.Cuda

open System
open System.Collections.Generic
open Marmotte.Core

/// Owned contiguous F32/F16 device tensors, stored-encoding weight uploads, F32/F16 Add,
/// SwiGLU, RMS normalization, row copies, RoPE/YaRN and embedding lookup of F32/F16/Q4_0/Q8_0
/// weight rows, using one CUDA stream per instance.
/// deviceIndex is a zero-based CUDA device ordinal (default 0), and must be non-negative.
/// Native loading and device initialization happen at the first non-empty allocation; construction,
/// empty tensors and an unused Synchronize require no CUDA installation. No CPU fallback is used.
/// Add, SwiGLU, RMS normalization, row copies, RoPE and zero-initialization may return before completion. Transfers, tensor disposal and
/// Synchronize wait for preceding work. Backend disposal waits and releases all remaining tensors
/// and weights.
/// UploadWeights copies the whole stored payload to device memory, so the WeightMatrix's source
/// may be released afterwards. Weight transfers use a staging array bounded to 4 MiB, without decoding;
/// see docs/cuda.md for costs.
/// Missing native libraries raise DllNotFoundException; CUDA failures raise InvalidOperationException
/// (OutOfMemoryException for allocation exhaustion). Unimplemented kernels and caches raise
/// NotImplementedException. Concurrent calls on an instance, including disposal, are unsupported.
[<Sealed>]
type CudaBackend(?deviceIndex: int) =
    let deviceIndex = defaultArg deviceIndex 0
    let owner = obj()
    let liveStorage = HashSet<ICudaStorage>(HashIdentity.Reference)
    let mutable session: CudaSession option = None
    let mutable disposed = false

    do
        if deviceIndex < 0 then
            invalidArg (nameof deviceIndex) "The CUDA device index must be non-negative."

    let checkDisposed () = ObjectDisposedException.ThrowIf(disposed, nameof CudaBackend)

    let getSession () =
        match session with
        | Some current -> current
        | None ->
            let created = new CudaSession(deviceIndex)
            session <- Some created
            created

    let synchronize () =
        match session with
        | Some current -> current.Synchronize()
        | None -> ()

    let pointerOf argumentName (tensor: ITensor) =
        ArgumentNullException.ThrowIfNull(tensor, argumentName)
        match tensor with
        | :? CudaTensor as tensor when Object.ReferenceEquals(tensor.Owner, owner) -> tensor.Pointer
        | _ -> invalidArg argumentName "The tensor must belong to this CUDA backend instance."

    let weightPointerOf argumentName (weights: IWeightTensor) =
        ArgumentNullException.ThrowIfNull(weights, argumentName)
        match weights with
        | :? CudaWeightTensor as tensor when Object.ReferenceEquals(tensor.Owner, owner) -> tensor.Pointer
        | _ -> invalidArg argumentName "The weights must belong to this CUDA backend instance."

    let release (storage: ICudaStorage) =
        try
            if storage.Pointer <> 0n then
                getSession().Free(storage.Pointer)
        finally
            liveStorage.Remove(storage) |> ignore

    let createTensor (shape: TensorShape) encoding =
        match encoding with
        | TensorEncoding.F32 | TensorEncoding.F16 -> ()
        | _ -> raise (NotSupportedException($"CUDA execution tensors do not support {encoding}."))

        let pointer = if shape.ElementCount = 0 then 0n else getSession().Allocate(shape.ElementCount, encoding)
        let tensor = new CudaTensor(owner, shape, encoding, pointer, release)
        liveStorage.Add(tensor) |> ignore
        tensor :> ITensor

    let requireSameShape operation (reference: ITensor) argumentName (tensor: ITensor) =
        if reference.Shape <> tensor.Shape then
            invalidArg argumentName $"{operation} requires identical shapes, not {reference.Shape} and {tensor.Shape}; broadcasting is not supported."

    let requireEncoding encoding argumentName (tensor: ITensor) =
        if tensor.Encoding <> encoding then
            invalidArg argumentName $"Expected an {encoding} tensor, not {tensor.Encoding}; implicit conversion is not supported."

    let notImplemented operation =
        checkDisposed ()
        raise (NotImplementedException($"CUDA {operation} is not implemented."))

    // Internal diagnostics verify stored bytes and native allocation accounting before weight
    // kernels exist. They never expose a device pointer through the public backend contract.
    member internal _.DownloadWeights(weights: IWeightTensor) =
        checkDisposed ()
        let pointer = weightPointerOf (nameof weights) weights
        let bytes = weights.Rows * (weights.Columns / weights.Encoding.BlockElements) * weights.Encoding.BlockBytes
        getSession().DownloadStored(pointer, bytes)

    member internal _.AllocatedBytes =
        checkDisposed ()
        match session with
        | Some current -> current.AllocatedBytes
        | None -> 0L

    interface ITensorBackend with
        member _.Backend = BackendKind.Cuda
        member this.Allocate(shape) =
            (this :> ITensorBackend).Allocate(shape, TensorEncoding.F32)

        member _.Allocate(shape, encoding) =
            checkDisposed ()
            ArgumentNullException.ThrowIfNull(shape)
            createTensor shape encoding

        member _.Upload(shape: TensorShape, values: ReadOnlyMemory<single>) =
            checkDisposed ()
            ArgumentNullException.ThrowIfNull(shape)
            if values.Length <> shape.ElementCount then
                invalidArg (nameof values) "Host value count must match the tensor shape."

            let tensor = createTensor shape TensorEncoding.F32
            try
                if values.Length > 0 then
                    getSession().Upload(pointerOf (nameof tensor) tensor, values.ToArray())
                tensor
            with _ ->
                // Preserve the upload failure if cleanup also encounters a device failure.
                try tensor.Dispose() with _ -> ()
                reraise ()

        member _.Upload(shape: TensorShape, values: ReadOnlyMemory<Half>) =
            checkDisposed ()
            ArgumentNullException.ThrowIfNull(shape)
            if values.Length <> shape.ElementCount then
                invalidArg (nameof values) "Host value count must match the tensor shape."

            let tensor = createTensor shape TensorEncoding.F16
            try
                if values.Length > 0 then
                    getSession().UploadHalf(pointerOf (nameof tensor) tensor, values)
                tensor
            with _ ->
                try tensor.Dispose() with _ -> ()
                reraise ()

        member _.UploadWeights(weights: WeightMatrix) =
            checkDisposed ()
            ArgumentNullException.ThrowIfNull(weights)
            let payload = weights.Payload
            let pointer = getSession().AllocateStored(weights.Encoding, int64 payload.Length)
            let tensor = new CudaWeightTensor(owner, weights.Name, weights.Encoding, weights.Rows, weights.Columns, pointer, release)
            liveStorage.Add(tensor) |> ignore
            try
                getSession().UploadStored(pointer, payload)
                tensor :> IWeightTensor
            with _ ->
                // Preserve the upload failure if cleanup also encounters a device failure.
                try (tensor :> IDisposable).Dispose() with _ -> ()
                reraise ()

        member _.AllocateKeyValueCache _ = notImplemented "key-value cache allocation"

        member _.Download(tensor, destination: Memory<single>) =
            checkDisposed ()
            let pointer = pointerOf (nameof tensor) tensor
            requireEncoding TensorEncoding.F32 (nameof tensor) tensor
            if destination.Length <> tensor.Shape.ElementCount then
                invalidArg (nameof destination) "Host value count must match the tensor shape."

            if destination.Length = 0 then
                synchronize ()
            else
                let values = Array.zeroCreate<single> destination.Length
                getSession().Download(pointer, values)
                values.AsMemory().CopyTo(destination)

        member _.Download(tensor, destination: Memory<Half>) =
            checkDisposed ()
            let pointer = pointerOf (nameof tensor) tensor
            requireEncoding TensorEncoding.F16 (nameof tensor) tensor
            if destination.Length <> tensor.Shape.ElementCount then
                invalidArg (nameof destination) "Host value count must match the tensor shape."

            if destination.Length = 0 then
                synchronize ()
            else
                getSession().DownloadHalf(pointer, destination)

        member _.Synchronize() =
            checkDisposed ()
            synchronize ()

    interface IAddKernel with
        member _.Add(left, right, destination) =
            checkDisposed ()
            let leftPointer = pointerOf (nameof left) left
            let rightPointer = pointerOf (nameof right) right
            let destinationPointer = pointerOf (nameof destination) destination
            requireSameShape "Add" left (nameof right) right
            requireSameShape "Add" left (nameof destination) destination
            requireEncoding left.Encoding (nameof right) right
            requireEncoding left.Encoding (nameof destination) destination

            if left.Shape.ElementCount > 0 then
                getSession().Add(leftPointer, rightPointer, destinationPointer, left.Shape.ElementCount)

    interface ISwiGluKernel with
        member _.SwiGlu(gate, up, destination) =
            checkDisposed ()
            let gatePointer = pointerOf (nameof gate) gate
            let upPointer = pointerOf (nameof up) up
            let destinationPointer = pointerOf (nameof destination) destination
            requireSameShape "SwiGLU" gate (nameof up) up
            requireSameShape "SwiGLU" gate (nameof destination) destination
            requireEncoding gate.Encoding (nameof up) up
            requireEncoding gate.Encoding (nameof destination) destination

            if gate.Shape.ElementCount > 0 then
                getSession().SwiGlu(gatePointer, upPointer, destinationPointer, gate.Shape.ElementCount)

    interface IRmsNormKernel with
        member _.RmsNorm(input, weight, epsilon, destination) =
            checkDisposed ()
            let inputPointer = pointerOf (nameof input) input
            let weightPointer = pointerOf (nameof weight) weight
            let destinationPointer = pointerOf (nameof destination) destination

            let shape = input.Shape
            if shape.Rank = 0 then
                invalidArg (nameof input) "RMS normalization requires a rank of at least 1; a scalar has no row to normalize."
            let columns = shape.Dimensions[shape.Rank - 1]
            if weight.Shape.Rank <> 1 || weight.Shape.Dimensions[0] <> columns then
                invalidArg (nameof weight) $"RMS normalization of {shape} requires weight of shape [{columns}], not {weight.Shape}."
            requireSameShape "RMS normalization" input (nameof destination) destination
            if not (Single.IsFinite epsilon) || epsilon <= 0.0f then
                raise (ArgumentOutOfRangeException(nameof epsilon, box epsilon, "Epsilon must be finite and positive."))
            requireEncoding input.Encoding (nameof weight) weight
            requireEncoding input.Encoding (nameof destination) destination

            let count = shape.ElementCount
            if count > 0 then
                getSession().RmsNorm(inputPointer, weightPointer, epsilon, destinationPointer, count / columns, columns)

    interface IEmbeddingKernel with
        member _.EmbeddingLookup(weights, tokenIds, destination) =
            checkDisposed ()
            let weightsPointer = weightPointerOf (nameof weights) weights
            let destinationPointer = pointerOf (nameof destination) destination

            let columns = weights.Columns
            let dimensions = destination.Shape.Dimensions
            if dimensions.Count <> 2 || dimensions[0] <> tokenIds.Length || dimensions[1] <> columns then
                invalidArg (nameof destination) $"Embedding lookup requires shape [{tokenIds.Length}; {columns}], not {destination.Shape}."
            requireEncoding TensorEncoding.F32 (nameof destination) destination

            // Reject unsupported stored encodings before submission, never decode another rule.
            match weights.Encoding with
            | TensorEncoding.F32 | TensorEncoding.F16 | TensorEncoding.Q4_0 | TensorEncoding.Q8_0 -> ()
            | encoding -> raise (NotSupportedException($"CUDA embedding lookup does not support {encoding} weights."))

            // Validate the entire batch first: a bad final token must not leave earlier rows written.
            let ids = tokenIds.Span
            for index = 0 to ids.Length - 1 do
                if ids[index] < 0 || ids[index] >= weights.Rows then
                    raise (ArgumentOutOfRangeException(nameof tokenIds, $"Token ID {ids[index]} at index {index} must be in [0, {weights.Rows})."))

            if destination.Shape.ElementCount > 0 then
                // The native call stages the IDs before returning, so the caller may reuse its
                // buffer immediately. Only the selected rows are decoded, on the GPU stream.
                let rowBytes = columns / weights.Encoding.BlockElements * weights.Encoding.BlockBytes
                getSession().EmbeddingLookup(weightsPointer, ids.ToArray(), destinationPointer,
                                              tokenIds.Length, weights.Rows, columns, rowBytes)

    interface IRowCopyKernel with
        member _.CopyRows(source, firstRow, destination) =
            checkDisposed ()
            let sourcePointer = pointerOf (nameof source) source
            let destinationPointer = pointerOf (nameof destination) destination

            let sourceShape = source.Shape.Dimensions
            let destinationShape = destination.Shape.Dimensions
            if sourceShape.Count <> 2 || destinationShape.Count <> 2 || sourceShape[1] <> destinationShape[1] then
                invalidArg (nameof destination) $"Row copies require rank-2 tensors with equal column counts, not {source.Shape} and {destination.Shape}."
            requireEncoding source.Encoding (nameof destination) destination

            let sourceRows = sourceShape[0]
            let rows = destinationShape[0]
            if firstRow < 0 || firstRow > sourceRows || rows > sourceRows - firstRow then
                raise (ArgumentOutOfRangeException(nameof firstRow, firstRow, $"Copying {rows} rows from row {firstRow} must fit the {sourceRows} source rows."))

            if destination.Shape.ElementCount > 0 then
                getSession().CopyRows(sourcePointer, sourceRows, firstRow, destinationPointer, rows, sourceShape[1])

    interface IMatVecKernel with
        member _.MatVec(_, _, _) = notImplemented "matrix-vector product"

    interface IMatMulKernel with
        member _.MatMul(_, _, _) = notImplemented "matrix-matrix product"

    interface IRopeKernel with
        member _.Rope(input, positions, embedding, destination) =
            checkDisposed ()
            ArgumentNullException.ThrowIfNull(embedding)
            let inputPointer = pointerOf (nameof input) input
            let destinationPointer = pointerOf (nameof destination) destination

            let shape = input.Shape
            if shape.Rank < 2 then
                invalidArg (nameof input) "RoPE requires a rank of at least 2 with tokens in the first dimension."
            let tokens = shape.Dimensions[0]
            if positions.Length <> tokens then
                invalidArg (nameof positions) $"RoPE requires {tokens} token positions, not {positions.Length}."
            let valuesPerToken =
                shape.Dimensions |> Seq.skip 1 |> Seq.fold (fun product dimension -> product * int64 dimension) 1L
            if valuesPerToken % int64 embedding.HeadDimension <> 0L then
                invalidArg (nameof input) $"RoPE requires whole heads of {embedding.HeadDimension} values per token, not {shape}."
            requireSameShape "RoPE" input (nameof destination) destination
            requireEncoding input.Encoding (nameof destination) destination

            let ids = positions.Span
            for index = 0 to ids.Length - 1 do
                if ids[index] < 0 then
                    raise (ArgumentOutOfRangeException(nameof positions, $"RoPE position at index {index} must be non-negative."))

            if shape.ElementCount > 0 then
                // Match the CPU reference: prepare double-precision angles once per token and
                // round cosine * magnitude and sine * magnitude to F32 before rotating heads.
                // The native call stages these coefficients before returning, so the caller may
                // reuse the positions buffer immediately. The GPU rotates values on its stream.
                let pairs = embedding.HeadDimension / 2
                let coefficients = Array.zeroCreate<single> (tokens * embedding.HeadDimension)
                let magnitude = float embedding.Magnitude
                for token = 0 to tokens - 1 do
                    for pair = 0 to pairs - 1 do
                        let angle = float ids[token] * embedding.Frequencies[pair]
                        let offset = token * embedding.HeadDimension + 2 * pair
                        coefficients[offset] <- single (Math.Cos(angle) * magnitude)
                        coefficients[offset + 1] <- single (Math.Sin(angle) * magnitude)

                let pairing =
                    match embedding.Pairing with
                    | RopePairing.AdjacentPairs -> 0
                    | RopePairing.SplitHalves -> 1
                let heads = int (valuesPerToken / int64 embedding.HeadDimension)
                getSession().Rope(inputPointer, destinationPointer, coefficients, tokens, heads,
                                  embedding.HeadDimension, pairing)

    interface IAttentionTemperatureKernel with
        member _.ApplyAttentionTemperature(_, _, _, _) = notImplemented "attention temperature kernel"

    interface IKeyValueCacheKernel with
        member _.StoreKeyValues(_, _, _, _, _) = notImplemented "key-value cache store"
        member _.ReadKeyValues(_, _, _, _, _) = notImplemented "key-value cache read"

    interface IAttentionKernel with
        member _.Attention(_, _, _, _, _) = notImplemented "grouped-query attention"

    interface IDisposable with
        member _.Dispose() =
            if not disposed then
                disposed <- true
                for storage in liveStorage do
                    storage.Invalidate()
                liveStorage.Clear()
                let closing = session
                session <- None
                match closing with
                | Some current -> (current :> IDisposable).Dispose()
                | None -> ()
