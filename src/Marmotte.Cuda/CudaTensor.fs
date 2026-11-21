namespace Marmotte.Cuda

open System
open Marmotte.Core

type internal CudaTensor(owner: obj, shape: TensorShape, encoding: TensorEncoding, pointer: nativeint, release: CudaTensor -> unit) =
    let mutable disposed = false

    member _.Owner = owner

    member this.Pointer =
        ObjectDisposedException.ThrowIf(disposed, this)
        pointer

    member _.Invalidate() = disposed <- true

    interface ICudaStorage with
        member this.Pointer = this.Pointer
        member this.Invalidate() = this.Invalidate()

    interface ITensor with
        member _.Shape = shape
        member _.Encoding = encoding
        member _.Backend = BackendKind.Cuda

    interface IDisposable with
        member this.Dispose() =
            if not disposed then
                try release this
                finally disposed <- true
