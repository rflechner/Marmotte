namespace Marmotte.Metal

open System
open System.Runtime.InteropServices
open Marmotte.Core

/// Private C ABI. SafeHandle keeps the context alive throughout each native call.
module private MetalApi =
    [<Literal>]
    let library = "marmotte_metal"

    // Status code of the native library; see native/MetalContext.h.
    [<Literal>]
    let outOfMemoryStatus = 2

    [<DllImport(library, CallingConvention = CallingConvention.Cdecl)>]
    extern nativeint marmotte_metal_last_error()

    [<DllImport(library, CallingConvention = CallingConvention.Cdecl)>]
    extern int marmotte_metal_create(nativeint& context)

    [<DllImport(library, CallingConvention = CallingConvention.Cdecl)>]
    extern int marmotte_metal_destroy(nativeint context)

    [<DllImport(library, CallingConvention = CallingConvention.Cdecl)>]
    extern int marmotte_metal_allocate(SafeHandle context, int count, nativeint& buffer)

    [<DllImport(library, CallingConvention = CallingConvention.Cdecl)>]
    extern int marmotte_metal_allocate_f16(SafeHandle context, int count, nativeint& buffer)

    [<DllImport(library, CallingConvention = CallingConvention.Cdecl)>]
    extern int marmotte_metal_upload(SafeHandle context, [<In>] single[] values, int count, nativeint& buffer)

    [<DllImport(library, CallingConvention = CallingConvention.Cdecl)>]
    extern int marmotte_metal_upload_f16(SafeHandle context, [<In>] uint16[] values, int count, nativeint& buffer)

    [<DllImport(library, CallingConvention = CallingConvention.Cdecl)>]
    extern int marmotte_metal_free(SafeHandle context, nativeint buffer)

    [<DllImport(library, CallingConvention = CallingConvention.Cdecl)>]
    extern int marmotte_metal_download(SafeHandle context, nativeint buffer, [<Out>] single[] values, int count)

    [<DllImport(library, CallingConvention = CallingConvention.Cdecl)>]
    extern int marmotte_metal_download_f16(SafeHandle context, nativeint buffer, [<Out>] uint16[] values, int count)

    [<DllImport(library, CallingConvention = CallingConvention.Cdecl)>]
    extern int marmotte_metal_add(SafeHandle context, nativeint left, nativeint right, nativeint destination, int count)

    [<DllImport(library, CallingConvention = CallingConvention.Cdecl)>]
    extern int marmotte_metal_add_f32_f16(SafeHandle context, nativeint left, nativeint right, nativeint destination, int count)

    [<DllImport(library, CallingConvention = CallingConvention.Cdecl)>]
    extern int marmotte_metal_synchronize(SafeHandle context)

    let check operation status =
        if status <> 0 then
            let detail = Marshal.PtrToStringUTF8(marmotte_metal_last_error ())
            let message = $"Metal {operation} failed ({status}): {detail}"
            if status = outOfMemoryStatus then
                raise (OutOfMemoryException(message))
            raise (InvalidOperationException(message))

/// Owns the native queue and every live buffer, including a finalization safety net.
type private MetalContextHandle(pointer: nativeint) =
    inherit SafeHandle(0n, true)
    do base.SetHandle(pointer)

    override this.IsInvalid = this.handle = 0n
    override this.ReleaseHandle() =
        MetalApi.marmotte_metal_destroy(this.handle) |> ignore
        true

/// Host arrays are marshalled only for synchronous transfers; no managed pointer escapes a call.
type internal MetalSession() =
    let context =
        if not Environment.Is64BitProcess || not (OperatingSystem.IsMacOS()) then
            raise (PlatformNotSupportedException("Metal requires a 64-bit macOS process."))

        let mutable pointer = 0n
        try
            MetalApi.marmotte_metal_create(&pointer) |> MetalApi.check "initialization"
        with :? DllNotFoundException as error ->
            raise (DllNotFoundException("The Marmotte Metal native library could not be loaded. Build and deploy marmotte_metal as described in docs/metal.md.", error))
        new MetalContextHandle(pointer)

    member _.Allocate(count, encoding) =
        let mutable buffer = 0n
        let status =
            match encoding with
            | TensorEncoding.F32 -> MetalApi.marmotte_metal_allocate(context, count, &buffer)
            | TensorEncoding.F16 -> MetalApi.marmotte_metal_allocate_f16(context, count, &buffer)
            | _ -> raise (NotSupportedException($"Metal execution tensors do not support {encoding}."))
        status |> MetalApi.check "allocation"
        buffer

    member _.Upload(values: single array) =
        let mutable buffer = 0n
        MetalApi.marmotte_metal_upload(context, values, values.Length, &buffer) |> MetalApi.check "upload"
        buffer

    /// Copy raw F16 bits through a two-byte primitive type supported by the P/Invoke marshaller.
    /// This safe span cast preserves NaN payloads and creates no F32 staging buffer.
    member _.UploadHalf(values: ReadOnlyMemory<Half>) =
        let bits = MemoryMarshal.Cast<Half, uint16>(values.Span).ToArray()
        let mutable buffer = 0n
        MetalApi.marmotte_metal_upload_f16(context, bits, bits.Length, &buffer) |> MetalApi.check "F16 upload"
        buffer

    member _.Free(buffer) = MetalApi.marmotte_metal_free(context, buffer) |> MetalApi.check "release"

    member _.Download(buffer, values: single array) =
        MetalApi.marmotte_metal_download(context, buffer, values, values.Length) |> MetalApi.check "download"

    member _.DownloadHalf(buffer, destination: Memory<Half>) =
        let bits = Array.zeroCreate<uint16> destination.Length
        MetalApi.marmotte_metal_download_f16(context, buffer, bits, bits.Length) |> MetalApi.check "F16 download"
        MemoryMarshal.Cast<uint16, Half>(bits.AsSpan()).CopyTo(destination.Span)

    member _.Add(left, right, destination, count) =
        MetalApi.marmotte_metal_add(context, left, right, destination, count) |> MetalApi.check "Add submission"

    member _.AddF32F16(left, right, destination, count) =
        MetalApi.marmotte_metal_add_f32_f16(context, left, right, destination, count) |> MetalApi.check "AddF32F16 submission"

    member _.Synchronize() = MetalApi.marmotte_metal_synchronize(context) |> MetalApi.check "synchronization"

    interface IDisposable with
        member _.Dispose() = context.Dispose()
