namespace Marmotte.Cuda

open System
open System.Runtime.InteropServices
open Marmotte.Core

/// Private C ABI. SafeHandle keeps the context alive throughout each native call.
module private CudaApi =
    [<Literal>]
    let library = "marmotte_cuda"

    // CUDA runtime error identifier; see the error reference linked in docs/cuda.md.
    [<Literal>]
    let memoryAllocationError = 2

    [<DllImport(library, CallingConvention = CallingConvention.Cdecl)>]
    extern nativeint marmotte_cuda_error(int status)

    [<DllImport(library, CallingConvention = CallingConvention.Cdecl)>]
    extern int marmotte_cuda_create(int device, nativeint& context)

    [<DllImport(library, CallingConvention = CallingConvention.Cdecl)>]
    extern int marmotte_cuda_destroy(nativeint context)

    [<DllImport(library, CallingConvention = CallingConvention.Cdecl)>]
    extern int marmotte_cuda_allocate(SafeHandle context, int count, nativeint& buffer)

    [<DllImport(library, CallingConvention = CallingConvention.Cdecl)>]
    extern int marmotte_cuda_allocate_f16(SafeHandle context, int count, nativeint& buffer)

    [<DllImport(library, CallingConvention = CallingConvention.Cdecl)>]
    extern int marmotte_cuda_free(SafeHandle context, nativeint buffer)

    [<DllImport(library, CallingConvention = CallingConvention.Cdecl)>]
    extern int marmotte_cuda_upload(SafeHandle context, nativeint buffer, [<In>] single[] values, int count)

    [<DllImport(library, CallingConvention = CallingConvention.Cdecl)>]
    extern int marmotte_cuda_download(SafeHandle context, nativeint buffer, [<Out>] single[] values, int count)

    [<DllImport(library, CallingConvention = CallingConvention.Cdecl)>]
    extern int marmotte_cuda_upload_f16(SafeHandle context, nativeint buffer, [<In>] uint16[] values, int count)

    [<DllImport(library, CallingConvention = CallingConvention.Cdecl)>]
    extern int marmotte_cuda_download_f16(SafeHandle context, nativeint buffer, [<Out>] uint16[] values, int count)

    [<DllImport(library, CallingConvention = CallingConvention.Cdecl)>]
    extern int marmotte_cuda_add(SafeHandle context, nativeint left, nativeint right, nativeint destination, int count)

    [<DllImport(library, CallingConvention = CallingConvention.Cdecl)>]
    extern int marmotte_cuda_synchronize(SafeHandle context)

    let check operation status =
        if status <> 0 then
            let detail = Marshal.PtrToStringUTF8(marmotte_cuda_error status)
            let message = $"CUDA {operation} failed ({status}): {detail}"
            if status = memoryAllocationError then
                raise (OutOfMemoryException(message))
            raise (InvalidOperationException(message))

/// Owns the native stream and every live device allocation, including a finalization safety net.
type private CudaContextHandle(pointer: nativeint) =
    inherit SafeHandle(0n, true)
    do base.SetHandle(pointer)

    override this.IsInvalid = this.handle = 0n
    override this.ReleaseHandle() =
        CudaApi.marmotte_cuda_destroy(this.handle) |> ignore
        true

/// Host arrays are marshalled only for synchronous transfers; no managed pointer escapes a call.
type internal CudaSession(deviceIndex: int) =
    let context =
        if not Environment.Is64BitProcess || not (OperatingSystem.IsWindows() || OperatingSystem.IsLinux()) then
            raise (PlatformNotSupportedException("CUDA requires a 64-bit Windows or Linux process."))

        let mutable pointer = 0n
        try
            CudaApi.marmotte_cuda_create(deviceIndex, &pointer) |> CudaApi.check "initialization"
        with :? DllNotFoundException as error ->
            raise (DllNotFoundException("The Marmotte CUDA native library could not be loaded. Build and deploy marmotte_cuda as described in docs/cuda.md.", error))
        new CudaContextHandle(pointer)

    member _.Allocate(count, encoding) =
        let mutable buffer = 0n
        let status =
            match encoding with
            | TensorEncoding.F32 -> CudaApi.marmotte_cuda_allocate(context, count, &buffer)
            | TensorEncoding.F16 -> CudaApi.marmotte_cuda_allocate_f16(context, count, &buffer)
            | _ -> raise (NotSupportedException($"CUDA execution tensors do not support {encoding}."))
        status |> CudaApi.check "allocation"
        buffer

    member _.Free(buffer) = CudaApi.marmotte_cuda_free(context, buffer) |> CudaApi.check "release"

    member _.Upload(buffer, values: single array) =
        CudaApi.marmotte_cuda_upload(context, buffer, values, values.Length) |> CudaApi.check "upload"

    member _.Download(buffer, values: single array) =
        CudaApi.marmotte_cuda_download(context, buffer, values, values.Length) |> CudaApi.check "download"

    /// Copy raw F16 bits through a two-byte primitive type supported by the P/Invoke marshaller.
    /// This safe span cast preserves NaN payloads and creates no F32 staging buffer.
    member _.UploadHalf(buffer, values: ReadOnlyMemory<Half>) =
        let bits = MemoryMarshal.Cast<Half, uint16>(values.Span).ToArray()
        CudaApi.marmotte_cuda_upload_f16(context, buffer, bits, bits.Length) |> CudaApi.check "F16 upload"

    member _.DownloadHalf(buffer, destination: Memory<Half>) =
        let bits = Array.zeroCreate<uint16> destination.Length
        CudaApi.marmotte_cuda_download_f16(context, buffer, bits, bits.Length) |> CudaApi.check "F16 download"
        MemoryMarshal.Cast<uint16, Half>(bits.AsSpan()).CopyTo(destination.Span)

    member _.Add(left, right, destination, count) =
        CudaApi.marmotte_cuda_add(context, left, right, destination, count) |> CudaApi.check "Add launch"

    member _.Synchronize() = CudaApi.marmotte_cuda_synchronize(context) |> CudaApi.check "synchronization"

    interface IDisposable with
        member _.Dispose() = context.Dispose()
