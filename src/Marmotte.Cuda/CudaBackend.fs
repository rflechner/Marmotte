namespace Marmotte.Cuda

open System
open Marmotte.Core

/// CUDA contract placeholder. Construction and disposal need no driver or device.
/// Allocation, transfers, weight upload, synchronization and Add raise NotImplementedException;
/// there is no CPU fallback.
/// Operations after disposal raise ObjectDisposedException.
[<Sealed>]
type CudaBackend() =
    let mutable disposed = false

    let notImplemented operation =
        if disposed then
            raise (ObjectDisposedException(nameof CudaBackend))
        raise (NotImplementedException($"CUDA {operation} is not implemented."))

    interface ITensorBackend with
        member _.Backend = BackendKind.Cuda
        member _.Allocate(_) = notImplemented "tensor allocation"
        member _.Upload(_, _) = notImplemented "tensor upload"
        member _.UploadWeights(_) = notImplemented "weight upload"
        member _.Download(_, _) = notImplemented "tensor download"
        member _.Synchronize() = notImplemented "synchronization"

    interface IAddKernel with
        member _.Add(_, _, _) = notImplemented "Add kernel"

    interface IDisposable with
        member _.Dispose() = disposed <- true
