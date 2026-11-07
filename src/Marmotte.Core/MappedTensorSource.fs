#nowarn "9" // Zero-copy spans over the mapped view use native pointers; the measurement that justifies this is recorded in docs/gguf.md.
#nowarn "3391" // Memory<byte> converts implicitly to ReadOnlyMemory<byte>.

namespace Marmotte.Core

open System
open System.Buffers
open System.Collections.Generic
open System.IO
open System.IO.MemoryMappedFiles
open System.Runtime.InteropServices
open Microsoft.FSharp.NativeInterop
open Microsoft.Win32.SafeHandles

/// One payload range of a memory-mapped view, exposed as ReadOnlyMemory without copying.
/// The view handle is reference-counted, so the mapping stays alive as long as this memory is
/// reachable, even after the source that created it is disposed; the base address is captured
/// once at construction because SafeHandle members may not be used after that disposal.
/// The pointer arithmetic is the only unsafe code in Core: it is confined to this type, the
/// offset and length were validated against the file size, and the measurement that justifies
/// reading mapped weights without copying is recorded in docs/gguf.md.
type private MappedPayloadMemory(handle: SafeMemoryMappedViewHandle, offset: int64, length: int) =
    inherit MemoryManager<byte>()

    let mutable referenceAdded = false

    let baseAddress =
        let mutable success = false
        handle.DangerousAddRef(&success)
        referenceAdded <- success
        handle.DangerousGetHandle()

    // Offsets add in nativeint so payloads beyond 2 GiB stay exact on 64-bit processes.
    let payloadPointer (elementIndex: int) =
        NativePtr.ofNativeInt<byte> (baseAddress + nativeint offset + nativeint elementIndex)

    override _.GetSpan() =
        Span<byte>(NativePtr.toVoidPtr (payloadPointer 0), length)

    override _.Pin(elementIndex: int) =
        new MemoryHandle(NativePtr.toVoidPtr (payloadPointer elementIndex))

    override _.Unpin() = ()

    override _.Dispose(_: bool) =
        if referenceAdded then
            handle.DangerousRelease()

/// Tensor source over one memory-mapped file, built by a format-specific source after it has
/// validated the file's index. Takes ownership of the stream once constructed and disposes it
/// together with the mapping. ReadPayload returns zero-copy memory over the mapped view, so
/// payloads never occupy managed heap and the operating system pages them in on demand and can
/// evict them under memory pressure; returned memory keeps the mapping alive and outlives the
/// source. A payload must fit in a single managed array. The file length is re-checked on every
/// read so that a file truncated after opening raises EndOfStreamException here instead of
/// faulting on a page past its end, but a truncation racing a read can still fault the process:
/// treat mapped model files as immutable.
///
/// Measured on 2026-11-07 against FileTensorSource on a synthetic 2.0 GiB GGUF (eight 256 MiB
/// Q4_K tensors plus one F32 vector, warm page cache, macOS Arm64): reading every payload by
/// copy took 177 ms and grew the GC heap by 2.2 GiB, while mapping the file took under 1 ms,
/// touching every page took 60 ms with no heap growth, and the resident pages are file-backed
/// and evictable. Copying from the mapping instead of reading the file handle measured worse
/// than both (701 ms, 4.4 GiB resident), so the zero-copy view is the adopted design. Full
/// numbers are in docs/gguf.md.
[<Sealed>]
type MappedTensorSource(file: FileStream, locations: FileTensorLocation array) =
    do
        ArgumentNullException.ThrowIfNull(file)
        ArgumentNullException.ThrowIfNull(locations)
        if not file.CanRead || not file.CanSeek then
            invalidArg (nameof file) "The tensor file must be readable and seekable."

    let fileLength = file.Length
    let locationsByName, descriptors = FileTensorLocations.validate fileLength locations

    // A zero-length file has no mapping to create.
    let mapping =
        if fileLength <= 0L then
            invalidArg (nameof file) "The tensor file is empty and cannot be memory-mapped."
        MemoryMappedFile.CreateFromFile(file, null, 0L, MemoryMappedFileAccess.Read, HandleInheritability.None, true)

    let view =
        try
            mapping.CreateViewAccessor(0L, 0L, MemoryMappedFileAccess.Read)
        with _ ->
            mapping.Dispose()
            reraise ()

    let viewHandle = view.SafeMemoryMappedViewHandle

    let mutable disposed = false

    interface ITensorSource with
        member _.Tensors = descriptors :> IReadOnlyList<TensorDescriptor>

        member _.TryFindTensor(name) =
            match locationsByName.TryGetValue(name) with
            | true, location -> Some location.Descriptor
            | false, _ -> None

        member this.ReadPayload(name) =
            ObjectDisposedException.ThrowIf(disposed, this)
            let location = FileTensorLocations.find locationsByName name
            let length = FileTensorLocations.payloadLength location.Descriptor
            if location.AbsoluteOffset + int64 length > file.Length then
                raise (EndOfStreamException($"Tensor '{name}' payload ends beyond the end of the file."))
            let memory = new MappedPayloadMemory(viewHandle, location.AbsoluteOffset, length)
            let payload: ReadOnlyMemory<byte> = memory.Memory
            payload

    interface IDisposable with
        member _.Dispose() =
            if not disposed then
                disposed <- true
                view.Dispose()
                mapping.Dispose()
                file.Dispose()
