module Marmotte.Backends.CudaWeightTests

open System
open System.Buffers
open System.Collections.Generic
open System.IO
open System.Runtime.InteropServices
open Marmotte.Core
open Marmotte.Cpu
open Marmotte.Cuda
open Marmotte.Backends.CudaTests
open Xunit

let private matrixFromMemory encoding rows columns (payload: ReadOnlyMemory<byte>) =
    let descriptor =
        { Name = "weights"
          Encoding = Some encoding
          SourceEncoding = string encoding
          Shape = [| uint64 rows; uint64 columns |]
          ElementCount = uint64 rows * uint64 columns
          ByteSize = Some(uint64 payload.Length) }
    use source =
        { new ITensorSource with
            member _.Tensors = [| descriptor |] :> IReadOnlyList<_>
            member _.TryFindTensor(_) = Some descriptor
            member _.ReadPayload(_) = payload
            member _.Dispose() = () }
    WeightMatrix.Read(source, descriptor.Name)

// These byte counts independently specify the stored layouts. Values include arbitrary scale
// bits and NaN representations: upload must preserve every bit without interpreting any of them.
let private formats =
    [ TensorEncoding.F32, 3, 12
      TensorEncoding.F16, 3, 6
      TensorEncoding.BF16, 3, 6
      TensorEncoding.Q4_0, 64, 36
      TensorEncoding.Q4_1, 64, 40
      TensorEncoding.Q5_0, 64, 44
      TensorEncoding.Q5_1, 64, 48
      TensorEncoding.Q8_0, 64, 68
      TensorEncoding.IQ4_NL, 64, 36
      TensorEncoding.Q2_K, 512, 168
      TensorEncoding.Q3_K, 512, 220
      TensorEncoding.Q4_K, 512, 288
      TensorEncoding.Q5_K, 512, 352
      TensorEncoding.Q6_K, 512, 420 ]

[<Fact>]
let ``CUDA weight disposal releases once and metadata survives release failures`` () =
    let mutable releases = 0
    let release (_: CudaWeightTensor) =
        releases <- releases + 1
        raise (InvalidOperationException("Release failed."))
    use tensor = new CudaWeightTensor(obj(), "weights", TensorEncoding.Q6_K, 2, 256, 123n, release)
    let weights = tensor :> IWeightTensor
    Assert.Throws<InvalidOperationException>(fun () -> weights.Dispose()) |> ignore
    weights.Dispose()
    Assert.Equal(1, releases)
    Assert.Throws<ObjectDisposedException>(fun () -> tensor.Pointer |> ignore) |> ignore
    Assert.Equal("weights", weights.Name)
    Assert.Equal(TensorEncoding.Q6_K, weights.Encoding)
    Assert.Equal(2, weights.Rows)
    Assert.Equal(256, weights.Columns)
    Assert.Equal(BackendKind.Cuda, weights.Backend)

[<Fact>]
let ``CUDA backend invalidation prevents another weight release`` () =
    let mutable releases = 0
    use tensor = new CudaWeightTensor(obj(), "weights", TensorEncoding.F32, 1, 1, 123n, fun _ -> releases <- releases + 1)
    (tensor :> ICudaStorage).Invalidate()
    (tensor :> IDisposable).Dispose()
    Assert.Equal(0, releases)
    Assert.Throws<ObjectDisposedException>(fun () -> tensor.Pointer |> ignore) |> ignore

[<CudaFact; Trait("Category", "CUDA")>]
let ``CUDA copies all fourteen stored encodings and frees each native allocation`` () =
    use backend = new CudaBackend()
    let device = backend :> ITensorBackend
    for encoding, columns, rowBytes in formats do
        let expected = Array.init (3 * rowBytes) (fun i -> byte ((i * 73 + 19) % 256))
        let host = Array.append [| 91uy |] (Array.append expected [| 92uy |])
        let matrix = matrixFromMemory encoding 3 columns (ReadOnlyMemory<byte>(host, 1, expected.Length))
        use weights = device.UploadWeights(matrix)
        Array.Fill(host, 0uy)
        GC.Collect()
        Assert.Equal<byte>(expected, backend.DownloadWeights(weights))
        Assert.Equal(int64 expected.Length, backend.AllocatedBytes)
        Assert.Equal(matrix.Name, weights.Name)
        Assert.Equal(encoding, weights.Encoding)
        Assert.Equal(3, weights.Rows)
        Assert.Equal(columns, weights.Columns)
        weights.Dispose()
        weights.Dispose()
        Assert.Equal(0L, backend.AllocatedBytes)
        Assert.Throws<ObjectDisposedException>(fun () -> backend.DownloadWeights(weights) |> ignore) |> ignore

[<CudaFact; Trait("Category", "CUDA")>]
let ``CUDA stored transfers preserve blocks crossing staging boundaries`` () =
    use backend = new CudaBackend()
    // One Q4_K block is 144 bytes: the 4 MiB staging boundary cuts through a block.
    for blocks in [ 29127; 29128; 58255 ] do
        let expected = Array.init (blocks * 144) (fun i -> byte ((i / 97 + i * 31) % 256))
        let matrix = matrixFromMemory TensorEncoding.Q4_K blocks 256 (ReadOnlyMemory<byte>(expected))
        use weights = (backend :> ITensorBackend).UploadWeights(matrix)
        Assert.Equal<byte>(expected, backend.DownloadWeights(weights))
        Assert.Equal(int64 expected.Length, backend.AllocatedBytes)

[<CudaFact; Trait("Category", "CUDA")>]
let ``CUDA copied weights survive closing and deleting their mapped source`` () =
    let path = Path.Combine(Path.GetTempPath(), $"marmotte-cuda-{Guid.NewGuid():N}.bin")
    let expected = Array.init 840 (fun i -> byte (i % 251))
    try
        File.WriteAllBytes(path, Array.append (Array.zeroCreate 32) expected)
        let descriptor =
            { Name = "mapped"
              Encoding = Some TensorEncoding.Q6_K
              SourceEncoding = "Q6_K"
              Shape = [| 2UL; 512UL |]
              ElementCount = 1024UL
              ByteSize = Some 840UL }
        use backend = new CudaBackend()
        let upload () =
            use source = new MappedTensorSource(File.OpenRead(path), [| { Descriptor = descriptor; AbsoluteOffset = 32L } |]) :> ITensorSource
            (backend :> ITensorBackend).UploadWeights(WeightMatrix.Read(source, "mapped"))
        use weights = upload ()
        File.Delete(path)
        Assert.Equal<byte>(expected, backend.DownloadWeights(weights))
    finally
        if File.Exists(path) then File.Delete(path)

type private FailingPayload(length: int) =
    inherit MemoryManager<byte>()
    let bytes = Array.zeroCreate<byte> length
    let mutable reads = 0
    override _.GetSpan() =
        reads <- reads + 1
        if reads > 1 then raise (IOException("The payload became unavailable."))
        bytes.AsSpan()
    override _.Pin(_) = raise (NotSupportedException())
    override _.Unpin() = ()
    override _.Dispose(_) = ()

[<CudaFact; Trait("Category", "CUDA")>]
let ``CUDA cleans up a partially uploaded weight while preserving other allocations`` () =
    use backend = new CudaBackend()
    let device = backend :> ITensorBackend
    use survivor = device.Upload(TensorShape.Create [| 1 |], ReadOnlyMemory<single>([| 7.0f |]))
    use payload = new FailingPayload(4 * 1024 * 1024 + 4)
    let matrix = matrixFromMemory TensorEncoding.F32 1 (1024 * 1024 + 1) (Memory<byte>.op_Implicit payload.Memory)
    let error = Assert.Throws<IOException>(fun () -> device.UploadWeights(matrix) |> ignore)
    Assert.Equal("The payload became unavailable.", error.Message)
    Assert.Equal(4L, backend.AllocatedBytes)
    let actual = Array.zeroCreate<single> 1
    device.Download(survivor, actual.AsMemory())
    Assert.Equal(7.0f, actual[0])

[<CudaFact; Trait("Category", "CUDA")>]
let ``CUDA weight ownership and disposal preserve another backend and queued work`` () =
    use backend = new CudaBackend()
    use other = new CudaBackend()
    use cpu = new CpuBackend() :> ITensorBackend
    let device = backend :> ITensorBackend
    let expected = Array.init 36 byte
    let matrix = matrixFromMemory TensorEncoding.Q4_0 1 64 (ReadOnlyMemory<byte>(expected))
    use weights = device.UploadWeights(matrix)
    use remaining = device.UploadWeights(matrix)
    use foreign = (other :> ITensorBackend).UploadWeights(matrix)
    use host = cpu.UploadWeights(matrix)
    for invalid in [ foreign; host ] do
        Assert.Throws<ArgumentException>(fun () -> backend.DownloadWeights(invalid) |> ignore) |> ignore
    Assert.Throws<ArgumentNullException>(fun () -> backend.DownloadWeights(Unchecked.defaultof<IWeightTensor>) |> ignore) |> ignore
    let shape = TensorShape.Create [| 65537 |]
    use tensor = device.Upload(shape, ReadOnlyMemory<single>(Array.create shape.ElementCount 3.0f))
    device.Add(tensor, tensor, tensor)
    weights.Dispose()
    let values = Array.zeroCreate<single> shape.ElementCount
    device.Download(tensor, values.AsMemory())
    Assert.All(values, fun value -> Assert.Equal(6.0f, value))
    device.Add(tensor, tensor, tensor)
    device.Dispose()
    device.Dispose()
    remaining.Dispose()
    Assert.Throws<ObjectDisposedException>(fun () -> (remaining :?> CudaWeightTensor).Pointer |> ignore) |> ignore
    Assert.Equal<byte>(expected, other.DownloadWeights(foreign))
    Assert.Throws<ObjectDisposedException>(fun () -> device.UploadWeights(matrix) |> ignore) |> ignore

module private Native =
    [<DllImport("marmotte_cuda", CallingConvention = CallingConvention.Cdecl)>]
    extern int marmotte_cuda_create(int device, nativeint& context)
    [<DllImport("marmotte_cuda", CallingConvention = CallingConvention.Cdecl)>]
    extern int marmotte_cuda_destroy(nativeint context)
    [<DllImport("marmotte_cuda", CallingConvention = CallingConvention.Cdecl)>]
    extern int marmotte_cuda_allocate_stored(nativeint context, int encoding, int64 bytes, nativeint& buffer)
    [<DllImport("marmotte_cuda", CallingConvention = CallingConvention.Cdecl)>]
    extern int marmotte_cuda_upload_stored_range(nativeint context, nativeint buffer, int64 offset, byte[] payload, int64 bytes)
    [<DllImport("marmotte_cuda", CallingConvention = CallingConvention.Cdecl)>]
    extern int marmotte_cuda_download_stored(nativeint context, nativeint buffer, [<Out>] byte[] payload, int64 bytes)
    [<DllImport("marmotte_cuda", CallingConvention = CallingConvention.Cdecl)>]
    extern int marmotte_cuda_add(nativeint context, nativeint left, nativeint right, nativeint destination, int count)
    [<DllImport("marmotte_cuda", CallingConvention = CallingConvention.Cdecl)>]
    extern int marmotte_cuda_free(nativeint context, nativeint buffer)

[<CudaFact; Trait("Category", "CUDA")>]
let ``CUDA native stored ranges reject overflow foreign pointers and execution tensor access`` () =
    let mutable context = 0n
    let mutable other = 0n
    Assert.Equal(0, Native.marmotte_cuda_create(0, &context))
    try
        Assert.Equal(0, Native.marmotte_cuda_create(0, &other))
        let mutable pointer = 0n
        for encoding, bytes in [ -1, 4L; 14, 4L; 0, 0L; 0, -1L ] do
            Assert.NotEqual(0, Native.marmotte_cuda_allocate_stored(context, encoding, bytes, &pointer))
            Assert.Equal(0n, pointer)
        Assert.Equal(0, Native.marmotte_cuda_allocate_stored(context, 0, 12L, &pointer))
        let expected = Array.init 12 (fun i -> byte (i + 1))
        Assert.Equal(0, Native.marmotte_cuda_upload_stored_range(context, pointer, 0L, expected, 12L))
        for offset, bytes in [ -1L, 1L; 0L, 0L; 0L, -1L; 12L, 1L; 1L, Int64.MaxValue; Int64.MaxValue, 1L ] do
            Assert.NotEqual(0, Native.marmotte_cuda_upload_stored_range(context, pointer, offset, expected, bytes))
        Assert.NotEqual(0, Native.marmotte_cuda_upload_stored_range(other, pointer, 0L, expected, 12L))
        for count in [ -1; 0; 3 ] do
            Assert.NotEqual(0, Native.marmotte_cuda_add(context, pointer, pointer, pointer, count))
        let actual = Array.create 12 255uy
        Assert.NotEqual(0, Native.marmotte_cuda_download_stored(context, pointer, actual, 11L))
        Assert.All(actual, fun value -> Assert.Equal(255uy, value))
        Assert.Equal(0, Native.marmotte_cuda_download_stored(context, pointer, actual, 12L))
        Assert.Equal<byte>(expected, actual)
        Assert.NotEqual(0, Native.marmotte_cuda_free(other, pointer))
        Assert.Equal(0, Native.marmotte_cuda_free(context, pointer))
        Assert.NotEqual(0, Native.marmotte_cuda_free(context, pointer))
    finally
        if other <> 0n then Assert.Equal(0, Native.marmotte_cuda_destroy(other))
        Assert.Equal(0, Native.marmotte_cuda_destroy(context))
