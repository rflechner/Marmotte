namespace Marmotte.Cuda

open System
open Marmotte.Core

/// CUDA contract placeholder. Construction and disposal need no driver or device.
/// Allocation, transfers, weight upload, synchronization and kernels raise NotImplementedException;
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

    interface IDisposable with
        member _.Dispose() = disposed <- true
