namespace Marmotte.Cuda

open System
open System.Collections.Generic
open Marmotte.Core

/// Device pointers never escape the assembly. Metadata survives disposal, just as on the CPU.
type private CudaTensor(owner: obj, shape: TensorShape, pointer: nativeint, release: CudaTensor -> unit) =
    let mutable disposed = false

    member _.Owner = owner

    member this.Pointer =
        ObjectDisposedException.ThrowIf(disposed, this)
        pointer

    member _.Invalidate() = disposed <- true

    interface ITensor with
        member _.Shape = shape
        member _.Encoding = TensorEncoding.F32
        member _.Backend = BackendKind.Cuda

    interface IDisposable with
        member this.Dispose() =
            if not disposed then
                try release this
                finally disposed <- true

/// Owned contiguous F32 device tensors and Add, using one CUDA stream per instance.
/// deviceIndex is a zero-based CUDA device ordinal (default 0), and must be non-negative.
/// Native loading and device initialization happen at the first non-empty allocation; construction,
/// empty tensors and an unused Synchronize require no CUDA installation. No CPU fallback is used.
/// Add and zero-initialization may return before completion. Transfers, tensor disposal and
/// Synchronize wait for preceding work. Backend disposal waits and releases all remaining tensors.
/// Host transfers use temporary managed arrays to avoid unsafe code; see docs/cuda.md for costs.
/// Missing native libraries raise DllNotFoundException; CUDA failures raise InvalidOperationException
/// (OutOfMemoryException for allocation exhaustion). Unimplemented kernels, weights and caches raise
/// NotImplementedException. Concurrent calls on an instance, including disposal, are unsupported.
[<Sealed>]
type CudaBackend(?deviceIndex: int) =
    let deviceIndex = defaultArg deviceIndex 0
    let owner = obj()
    let liveTensors = HashSet<CudaTensor>(HashIdentity.Reference)
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

    let release (tensor: CudaTensor) =
        try
            if tensor.Pointer <> 0n then
                getSession().Free(tensor.Pointer)
        finally
            liveTensors.Remove(tensor) |> ignore

    let createTensor (shape: TensorShape) =
        let pointer = if shape.ElementCount = 0 then 0n else getSession().Allocate(shape.ElementCount)
        let tensor = new CudaTensor(owner, shape, pointer, release)
        liveTensors.Add(tensor) |> ignore
        tensor :> ITensor

    let requireSameShape (reference: ITensor) argumentName (tensor: ITensor) =
        if reference.Shape <> tensor.Shape then
            invalidArg argumentName $"Add requires identical shapes, not {reference.Shape} and {tensor.Shape}; broadcasting is not supported."

    let notImplemented operation =
        checkDisposed ()
        raise (NotImplementedException($"CUDA {operation} is not implemented."))

    interface ITensorBackend with
        member _.Backend = BackendKind.Cuda
        member _.Allocate(shape) =
            checkDisposed ()
            ArgumentNullException.ThrowIfNull(shape)
            createTensor shape

        member _.Upload(shape, values) =
            checkDisposed ()
            ArgumentNullException.ThrowIfNull(shape)
            if values.Length <> shape.ElementCount then
                invalidArg (nameof values) "Host value count must match the tensor shape."

            let tensor = createTensor shape
            try
                if values.Length > 0 then
                    getSession().Upload(pointerOf (nameof tensor) tensor, values.ToArray())
                tensor
            with _ ->
                // Preserve the upload failure if cleanup also encounters a device failure.
                try tensor.Dispose() with _ -> ()
                reraise ()

        member _.UploadWeights(_) = notImplemented "weight upload"
        member _.AllocateKeyValueCache(_) = notImplemented "key-value cache allocation"

        member _.Download(tensor, destination) =
            checkDisposed ()
            let pointer = pointerOf (nameof tensor) tensor
            if destination.Length <> tensor.Shape.ElementCount then
                invalidArg (nameof destination) "Host value count must match the tensor shape."

            if destination.Length = 0 then
                synchronize ()
            else
                let values = Array.zeroCreate<single> destination.Length
                getSession().Download(pointer, values)
                values.AsMemory().CopyTo(destination)

        member _.Synchronize() =
            checkDisposed ()
            synchronize ()

    interface IAddKernel with
        member _.Add(left, right, destination) =
            checkDisposed ()
            let leftPointer = pointerOf (nameof left) left
            let rightPointer = pointerOf (nameof right) right
            let destinationPointer = pointerOf (nameof destination) destination
            requireSameShape left (nameof right) right
            requireSameShape left (nameof destination) destination

            if left.Shape.ElementCount > 0 then
                getSession().Add(leftPointer, rightPointer, destinationPointer, left.Shape.ElementCount)

    interface ISwiGluKernel with
        member _.SwiGlu(_, _, _) = notImplemented "SwiGLU kernel"

    interface IRmsNormKernel with
        member _.RmsNorm(_, _, _, _) = notImplemented "RMS normalization kernel"

    interface IEmbeddingKernel with
        member _.EmbeddingLookup(_, _, _) = notImplemented "embedding lookup kernel"

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

    interface IDisposable with
        member _.Dispose() =
            if not disposed then
                disposed <- true
                try
                    synchronize ()
                finally
                    for tensor in liveTensors do
                        tensor.Invalidate()
                    liveTensors.Clear()
                    match session with
                    | Some current ->
                        (current :> IDisposable).Dispose()
                        session <- None
                    | None -> ()
