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

    /// The stored rows that future kernels decode or multiply.
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

/// Safe, scalar CPU reference implementation of the tensor and Add contracts.
/// Owns no native resources. Each instance accepts only tensors that it created, and tracks
/// the live ones so that disposing the backend releases them. Operations complete before
/// returning, which satisfies the ordering contract; Synchronize has nothing to wait for.
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

            if left.Shape <> right.Shape then
                invalidArg (nameof right) "Add requires identical input shapes; broadcasting is not supported."
            if left.Shape <> destination.Shape then
                invalidArg (nameof destination) "Add destination must have the same shape as its inputs."

            for index = 0 to destinationValues.Length - 1 do
                destinationValues[index] <- leftValues[index] + rightValues[index]

    interface IDisposable with
        member _.Dispose() =
            if not disposed then
                disposed <- true
                for storage in liveStorage do
                    storage.Release()
                liveStorage.Clear()
