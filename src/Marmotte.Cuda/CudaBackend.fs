namespace Marmotte.Cuda

open System
open System.Collections.Generic
open Marmotte.Core

/// Owned contiguous F32/F16 device tensors, stored-encoding weight uploads, and F32/F16 Add and
/// SwiGLU, using one CUDA stream per instance.
/// deviceIndex is a zero-based CUDA device ordinal (default 0), and must be non-negative.
/// Native loading and device initialization happen at the first non-empty allocation; construction,
/// empty tensors and an unused Synchronize require no CUDA installation. No CPU fallback is used.
/// Add, SwiGLU and zero-initialization may return before completion. Transfers, tensor disposal and
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
        ArgumentNullException.ThrowIfNull(weights)
        match weights with
        | :? CudaWeightTensor as tensor when Object.ReferenceEquals(tensor.Owner, owner) ->
            let pointer = tensor.Pointer
            let bytes = weights.Rows * (weights.Columns / weights.Encoding.BlockElements) * weights.Encoding.BlockBytes
            getSession().DownloadStored(pointer, bytes)
        | _ -> invalidArg (nameof weights) "The weights must belong to this CUDA backend instance."

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
        member _.RmsNorm(_, _, _, _) = notImplemented "RMS normalization kernel"

    interface IEmbeddingKernel with
        member _.EmbeddingLookup(_, _, _) = notImplemented "embedding lookup kernel"

    interface IRowCopyKernel with
        member _.CopyRows(_, _, _) = notImplemented "row copy"

    interface IMatVecKernel with
        member _.MatVec(_, _, _) = notImplemented "matrix-vector product"

    interface IMatMulKernel with
        member _.MatMul(_, _, _) = notImplemented "matrix-matrix product"

    interface IRopeKernel with
        member _.Rope(_, _, _, _) = notImplemented "RoPE kernel"

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
