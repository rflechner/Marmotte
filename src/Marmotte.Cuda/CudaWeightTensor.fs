namespace Marmotte.Cuda

open System
open Marmotte.Core

/// A weight matrix copied to device memory in its stored encoding, never decoded as a whole.
/// Unlike the CPU backend, the payload is copied at upload time, so the tensor source that
/// produced the WeightMatrix need not stay alive while this tensor is in use.
type internal CudaWeightTensor(owner: obj, name: string, encoding: TensorEncoding, rows: int, columns: int, pointer: nativeint, release: CudaWeightTensor -> unit) =
    let mutable disposed = false

    member _.Owner = owner

    member this.Pointer =
        ObjectDisposedException.ThrowIf(disposed, this)
        pointer

    member _.Invalidate() = disposed <- true

    interface ICudaStorage with
        member this.Pointer = this.Pointer
        member this.Invalidate() = this.Invalidate()

    interface IWeightTensor with
        member _.Name = name
        member _.Encoding = encoding
        member _.Rows = rows
        member _.Columns = columns
        member _.Backend = BackendKind.Cuda

    interface IDisposable with
        member this.Dispose() =
            if not disposed then
                try release this
                finally disposed <- true
