namespace Marmotte.Cpu

open System
open System.Collections.Generic
open System.Numerics
open System.Runtime.InteropServices
open Marmotte.Core

/// Element-wise F32 kernels over equal-length host spans, safe and vectorized with Vector<T>.
/// Each processes Vector<single>.Count lanes at a time (4 with ARM64 NEON, 8 with AVX2), then the
/// remaining elements one by one. Destination may be exactly an input, but must not partially
/// overlap one: each vector is read before the same positions are written.
module private CpuKernels =
    let private requireSameLength (first: ReadOnlySpan<single>) (second: ReadOnlySpan<single>) (destination: Span<single>) =
        if first.Length <> destination.Length || second.Length <> destination.Length then
            invalidArg (nameof destination) "Element-wise kernel spans must have the same length."

    /// destination[i] = left[i] + right[i]. Each lane performs the same IEEE single-precision
    /// addition as a scalar loop, so vector lanes and the tail give identical results.
    let add (left: ReadOnlySpan<single>, right: ReadOnlySpan<single>, destination: Span<single>) =
        requireSameLength left right destination

        let mutable firstScalar = 0
        if Vector.IsHardwareAccelerated then
            let leftVectors = MemoryMarshal.Cast<single, Vector<single>>(left)
            let rightVectors = MemoryMarshal.Cast<single, Vector<single>>(right)
            let destinationVectors = MemoryMarshal.Cast<single, Vector<single>>(destination)
            for index = 0 to destinationVectors.Length - 1 do
                destinationVectors[index] <- leftVectors[index] + rightVectors[index]
            firstScalar <- destinationVectors.Length * Vector<single>.Count

        for index = firstScalar to destination.Length - 1 do
            destination[index] <- left[index] + right[index]

    /// destination[i] = silu(gate[i]) × up[i], with silu(x) = x / (1 + e^-x). Vector lanes use
    /// Vector.Exp and the tail uses MathF.Exp; measured on ARM64, both stay within 2.2e-7 relative
    /// error of a double-precision SiLU above x = -87, and they differ by at most 4 units in the
    /// last place.
    let swiGlu (gate: ReadOnlySpan<single>, up: ReadOnlySpan<single>, destination: Span<single>) =
        requireSameLength gate up destination

        let mutable firstScalar = 0
        if Vector.IsHardwareAccelerated then
            let gateVectors = MemoryMarshal.Cast<single, Vector<single>>(gate)
            let upVectors = MemoryMarshal.Cast<single, Vector<single>>(up)
            let destinationVectors = MemoryMarshal.Cast<single, Vector<single>>(destination)
            for index = 0 to destinationVectors.Length - 1 do
                let x = gateVectors[index]
                let silu = x / (Vector<single>.One + Vector.Exp(-x))
                destinationVectors[index] <- silu * upVectors[index]
            firstScalar <- destinationVectors.Length * Vector<single>.Count

        for index = firstScalar to destination.Length - 1 do
            let x = gate[index]
            let silu = x / (1.0f + MathF.Exp(-x))
            destination[index] <- silu * up[index]

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

/// Safe CPU reference implementation of the tensor, Add and SwiGLU contracts, vectorized with Vector<T>.
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

    /// Element-wise kernels reject broadcasting: every tensor must have the reference's shape.
    let requireSameShape (operation: string) (reference: ITensor) argumentName (tensor: ITensor) =
        if reference.Shape <> tensor.Shape then
            invalidArg argumentName $"{operation} requires identical shapes, not {reference.Shape} and {tensor.Shape}; broadcasting is not supported."

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

            requireSameShape "Add" left (nameof right) right
            requireSameShape "Add" left (nameof destination) destination

            CpuKernels.add (ReadOnlySpan<single>(leftValues), ReadOnlySpan<single>(rightValues), Span<single>(destinationValues))

    interface ISwiGluKernel with
        member this.SwiGlu(gate, up, destination) =
            ObjectDisposedException.ThrowIf(disposed, this)
            let gateValues = valuesOf (nameof gate) gate
            let upValues = valuesOf (nameof up) up
            let destinationValues = valuesOf (nameof destination) destination

            requireSameShape "SwiGLU" gate (nameof up) up
            requireSameShape "SwiGLU" gate (nameof destination) destination

            CpuKernels.swiGlu (ReadOnlySpan<single>(gateValues), ReadOnlySpan<single>(upValues), Span<single>(destinationValues))

    interface IDisposable with
        member _.Dispose() =
            if not disposed then
                disposed <- true
                for storage in liveStorage do
                    storage.Release()
                liveStorage.Clear()
