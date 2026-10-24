namespace Marmotte.Core

open System
open System.Collections.Generic
open System.IO
open Microsoft.Win32.SafeHandles

/// Where a tensor's payload starts in a file.
type FileTensorLocation =
    { Descriptor: TensorDescriptor
      /// Absolute position of the payload in the file.
      AbsoluteOffset: int64 }

/// Tensor source over one open file, built by a format-specific source after it has validated
/// the file's index. Takes ownership of the stream once constructed and disposes it.
/// Each ReadPayload call copies the payload into a new managed array with positional reads,
/// so concurrent reads are safe and returned memory outlives the source. A payload must fit
/// in a single managed array. Memory mapping can replace the copy behind the same
/// ITensorSource contract once a measurement justifies it.
[<Sealed>]
type FileTensorSource(file: FileStream, locations: FileTensorLocation array) =
    do
        ArgumentNullException.ThrowIfNull(file)
        ArgumentNullException.ThrowIfNull(locations)
        if not file.CanRead || not file.CanSeek then
            invalidArg (nameof file) "The tensor file must be readable and seekable."

    let fileLength = file.Length

    let isWithinFile (location: FileTensorLocation) =
        if location.AbsoluteOffset < 0L || location.AbsoluteOffset > fileLength then
            false
        else
            let remaining = uint64 (fileLength - location.AbsoluteOffset)
            location.Descriptor.ByteSize |> Option.forall (fun size -> size <= remaining)

    let locationsByName =
        let entries = Dictionary<string, FileTensorLocation>(StringComparer.Ordinal)
        for location in locations do
            let name = location.Descriptor.Name
            if not (entries.TryAdd(name, location)) then
                invalidArg (nameof locations) $"Duplicate tensor name '{name}'."
            if not (isWithinFile location) then
                invalidArg (nameof locations) $"Tensor '{name}' payload lies outside the file."
        entries

    let descriptors = locations |> Array.map (fun location -> location.Descriptor)

    let mutable disposed = false

    let findLocation name =
        match locationsByName.TryGetValue(name) with
        | true, location -> location
        | false, _ -> raise (KeyNotFoundException($"Tensor '{name}' does not exist."))

    let payloadLength (descriptor: TensorDescriptor) =
        match descriptor.Encoding, descriptor.ByteSize with
        | None, _ ->
            raise (InvalidDataException(
                $"Tensor '{descriptor.Name}' uses {descriptor.SourceEncoding}, which Marmotte does not represent at runtime."))
        | Some _, None ->
            raise (InvalidDataException($"Tensor '{descriptor.Name}' has an unknown payload size."))
        | Some _, Some size when size > uint64 Array.MaxLength ->
            raise (InvalidDataException(
                $"Tensor '{descriptor.Name}' payload ({size} bytes) exceeds the managed array limit ({Array.MaxLength} bytes)."))
        | Some _, Some size -> int size

    /// RandomAccess.Read may return fewer bytes than requested; zero means the file was truncated.
    let readExactlyAt (handle: SafeFileHandle) (destination: Span<byte>) (position: int64) (name: string) =
        let mutable filled = 0
        while filled < destination.Length do
            let count = RandomAccess.Read(handle, destination.Slice(filled), position + int64 filled)
            if count = 0 then
                raise (EndOfStreamException($"Tensor '{name}' payload ends beyond the end of the file."))
            filled <- filled + count

    interface ITensorSource with
        member _.Tensors = descriptors :> IReadOnlyList<TensorDescriptor>

        member _.TryFindTensor(name) =
            match locationsByName.TryGetValue(name) with
            | true, location -> Some location.Descriptor
            | false, _ -> None

        member this.ReadPayload(name) =
            ObjectDisposedException.ThrowIf(disposed, this)
            let location = findLocation name
            let payload = Array.zeroCreate<byte> (payloadLength location.Descriptor)
            readExactlyAt file.SafeFileHandle (payload.AsSpan()) location.AbsoluteOffset name
            ReadOnlyMemory<byte>(payload)

    interface IDisposable with
        member _.Dispose() =
            if not disposed then
                disposed <- true
                file.Dispose()
