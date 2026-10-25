namespace Marmotte.Metal

open System
open Marmotte.Core

/// Metal contract placeholder. Construction and disposal need no driver or device.
/// Allocation, transfers, weight upload, synchronization and kernels raise NotImplementedException;
/// there is no CPU fallback.
/// Operations after disposal raise ObjectDisposedException.
[<Sealed>]
type MetalBackend() =
    let mutable disposed = false

    let notImplemented operation =
        if disposed then
            raise (ObjectDisposedException(nameof MetalBackend))
        raise (NotImplementedException($"Metal {operation} is not implemented."))

    interface ITensorBackend with
        member _.Backend = BackendKind.Metal
        member _.Allocate(_) = notImplemented "tensor allocation"
        member _.Upload(_, _) = notImplemented "tensor upload"
        member _.UploadWeights(_) = notImplemented "weight upload"
        member _.Download(_, _) = notImplemented "tensor download"
        member _.Synchronize() = notImplemented "synchronization"

    interface IAddKernel with
        member _.Add(_, _, _) = notImplemented "Add kernel"

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

    interface IDisposable with
        member _.Dispose() = disposed <- true
