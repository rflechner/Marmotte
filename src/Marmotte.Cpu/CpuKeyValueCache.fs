namespace Marmotte.Cpu

open System
open Marmotte.Core

/// Allocates the whole capacity up front: one byte array of Capacity encoded rows per layer for
/// keys and another for values, with position p at row p. The stored position count of each layer
/// is the only state besides the bytes.
type internal CpuKeyValueCache(owner: obj, layout: KeyValueCacheLayout, onDispose: ICpuStorage -> unit) =
    let allocateRows (rowBytes: int) =
        Array.init layout.Layers (fun _ -> Array.zeroCreate<byte> (layout.Capacity * rowBytes))

    let mutable keyRows = allocateRows layout.KeyRowBytes
    let mutable valueRows = allocateRows layout.ValueRowBytes
    let storedPositions = Array.zeroCreate<int> layout.Layers
    let mutable disposed = false

    member _.Owner = owner

    /// Encoded keys of a layer that StoredPositions has already validated.
    member this.KeyRows(layer: int) =
        ObjectDisposedException.ThrowIf(disposed, this)
        keyRows[layer]

    /// Encoded values of a layer that StoredPositions has already validated.
    member this.ValueRows(layer: int) =
        ObjectDisposedException.ThrowIf(disposed, this)
        valueRows[layer]

    member _.SetStoredPositions(layer: int, count: int) = storedPositions[layer] <- count

    interface ICpuStorage with
        member _.Release() =
            disposed <- true
            keyRows <- Array.empty
            valueRows <- Array.empty

    interface IKeyValueCache with
        member _.Layout = layout
        member _.Backend = BackendKind.Cpu

        member this.StoredPositions(layer) =
            ObjectDisposedException.ThrowIf(disposed, this)
            if layer < 0 || layer >= layout.Layers then
                raise (ArgumentOutOfRangeException(nameof layer, box layer, $"The layer must be in [0, {layout.Layers})."))
            storedPositions[layer]

    interface IDisposable with
        member this.Dispose() =
            if not disposed then
                (this :> ICpuStorage).Release()
                onDispose this
