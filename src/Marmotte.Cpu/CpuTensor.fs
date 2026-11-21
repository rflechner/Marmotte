namespace Marmotte.Cpu

open System
open Marmotte.Core

/// Execution storage retains its actual element width; F16 never owns an F32 shadow buffer.
[<RequireQualifiedAccess>]
type internal CpuValues =
    | F32 of single array
    | F16 of Half array

    member this.Encoding =
        match this with
        | CpuValues.F32 _ -> TensorEncoding.F32
        | CpuValues.F16 _ -> TensorEncoding.F16

/// The owned buffer is private to this backend assembly; every public transfer makes a copy.
type internal CpuTensor(owner: obj, shape: TensorShape, values: CpuValues, onDispose: ICpuStorage -> unit) =
    let encoding = values.Encoding
    let mutable retained = Some values

    member _.Owner = owner

    member this.Values =
        match retained with
        | Some values -> values
        | None -> raise (ObjectDisposedException(this.GetType().FullName))

    interface ICpuStorage with
        member _.Release() = retained <- None

    interface ITensor with
        member _.Shape = shape
        member _.Encoding = encoding
        member _.Backend = BackendKind.Cpu

    interface IDisposable with
        member this.Dispose() =
            if retained.IsSome then
                (this :> ICpuStorage).Release()
                onDispose this
