module Marmotte.Backends.CudaEmbeddingTests

open System
open System.Collections.Generic
open System.Runtime.InteropServices
open Marmotte.Core
open Marmotte.Cpu
open Marmotte.Cuda
open Marmotte.Backends.CudaTests
open Xunit

/// Serves one embedding matrix from memory, to build a WeightMatrix without a file format.
type private EmbeddingSource(encoding: TensorEncoding, rows: int, columns: int, payload: byte array) =
    let descriptor =
        { Name = "token_embd.weight"
          Encoding = Some encoding
          SourceEncoding = string encoding
          Shape = [| uint64 rows; uint64 columns |]
          ElementCount = uint64 rows * uint64 columns
          ByteSize = Some (uint64 payload.Length) }

    interface ITensorSource with
        member _.Tensors = [| descriptor |] :> IReadOnlyList<TensorDescriptor>
        member _.TryFindTensor(name) = if name = descriptor.Name then Some descriptor else None
        member _.ReadPayload(_) = ReadOnlyMemory<byte>(payload)

    interface IDisposable with
        member _.Dispose() = ()

let private matrixOf encoding rows columns payload =
    use source = new EmbeddingSource(encoding, rows, columns, payload)
    WeightMatrix.Read(source, "token_embd.weight")

let private download (backend: ITensorBackend) (tensor: ITensor) =
    let values = Array.zeroCreate<single> tensor.Shape.ElementCount
    backend.Download(tensor, values.AsMemory())
    values

let private cpuLookup (matrix: WeightMatrix) (ids: int array) =
    use cpu = new CpuBackend() :> ITensorBackend
    use weights = cpu.UploadWeights(matrix)
    use destination = cpu.Allocate(TensorShape.Create [| ids.Length; matrix.Columns |])
    cpu.EmbeddingLookup(weights, ReadOnlyMemory<int>(ids), destination)
    download cpu destination

// Hand-packed fixtures, independent of Marmotte's encoder and decoder. F16 scales are stored
// little-endian: 0x3400 = 0.25, 0x3800 = 0.5, 0x3C00 = 1.0, 0xC000 = -2.0.
let private f32Payload =
    Convert.FromHexString("0000803F000020C00000003E000080400000003F000040C0") // rows [1; -2.5; 0.125] and [4; 0.5; -3]

let private f16Payload =
    Convert.FromHexString("003C00C100300044003800C2")

// Two 64-value rows of two Q4_0 blocks each: low nibbles encode -8..7, high nibbles 7..-8.
let private q4_0Payload =
    let quants = Convert.FromHexString("F0E1D2C3B4A5968778695A4B3C2D1E0F")
    let first = Array.append [| 0x00uy; 0x38uy |] quants // scale 0.5
    let second = Array.append [| 0x00uy; 0xC0uy |] quants // scale -2
    Array.concat [ first; second; second; first ]

// Two 64-value rows of two Q8_0 blocks each: signed bytes -128, -1, 0, 1, 127, -64, 64, 32.
let private q8_0Payload =
    let quants = Array.replicate 4 (Convert.FromHexString("80FF00017FC04020")) |> Array.concat
    let first = Array.append [| 0x00uy; 0x34uy |] quants // scale 0.25
    let second = Array.append [| 0x00uy; 0xC0uy |] quants // scale -2
    Array.concat [ first; second; second; first ]

[<Fact>]
let ``CUDA embedding lookup validates null and disposed arguments without a driver`` () =
    use backend = new CudaBackend() :> ITensorBackend
    let ids = ReadOnlyMemory<int>([| 0 |])
    Assert.Throws<ArgumentNullException>(fun () ->
        backend.EmbeddingLookup(Unchecked.defaultof<IWeightTensor>, ids, Unchecked.defaultof<ITensor>)) |> ignore
    backend.Dispose()
    Assert.Throws<ObjectDisposedException>(fun () ->
        backend.EmbeddingLookup(Unchecked.defaultof<IWeightTensor>, ids, Unchecked.defaultof<ITensor>)) |> ignore

[<CudaFact>]
[<Trait("Category", "CUDA")>]
let ``CUDA embedding lookup matches CPU for F32 F16 Q4_0 and Q8_0 with repeated and reordered IDs`` () =
    use backend = new CudaBackend()
    let device = backend :> ITensorBackend
    let ids = [| 1; 0; 1; 0; 1 |]
    for encoding, columns, payload in [ (TensorEncoding.F32, 3, f32Payload);
                                       (TensorEncoding.F16, 3, f16Payload);
                                       (TensorEncoding.Q4_0, 64, q4_0Payload);
                                       (TensorEncoding.Q8_0, 64, q8_0Payload) ] do
        let matrix = matrixOf encoding 2 columns payload
        let original = Array.copy payload
        let expected = cpuLookup matrix ids
        use weights = device.UploadWeights(matrix)
        use destination = device.Allocate(TensorShape.Create [| ids.Length; columns |])
        device.EmbeddingLookup(weights, ReadOnlyMemory<int>(ids), destination)
        Assert.Equal<single>(expected, download device destination)
        // Only selected rows are decoded; the stored payload is unchanged.
        Assert.Equal<byte>(original, backend.DownloadWeights(weights))

        // A single-token generation step still has a two-dimensional output.
        use single = device.Allocate(TensorShape.Create [| 1; columns |])
        device.EmbeddingLookup(weights, ReadOnlyMemory<int>([| 0 |]), single)
        Assert.Equal<single>(cpuLookup matrix [| 0 |], download device single)

[<CudaFact>]
[<Trait("Category", "CUDA")>]
let ``CUDA embedding lookup decodes a hand-specified Q8_0 block exactly`` () =
    use backend = new CudaBackend()
    let device = backend :> ITensorBackend
    let block = Array.zeroCreate<byte> 34
    block[0] <- 0x00uy
    block[1] <- 0x34uy // F16 scale 0.25
    block[2] <- 0xF8uy // -8
    block[3] <- 0x00uy // 0
    block[4] <- 0x04uy // 4
    block[5] <- 0x07uy // 7
    let matrix = matrixOf TensorEncoding.Q8_0 1 32 block
    use weights = device.UploadWeights(matrix)
    use destination = device.Allocate(TensorShape.Create [| 1; 32 |])
    device.EmbeddingLookup(weights, ReadOnlyMemory<int>([| 0 |]), destination)
    let actual = download device destination
    Assert.Equal<single>([| -2.0f; 0.0f; 1.0f; 1.75f |], actual[..3])
    Assert.Equal<single>(Array.zeroCreate 28, actual[4..])

[<CudaFact>]
[<Trait("Category", "CUDA")>]
let ``CUDA embedding lookup copies IDs before returning and preserves stream order`` () =
    use backend = new CudaBackend()
    let device = backend :> ITensorBackend
    let matrix = matrixOf TensorEncoding.F32 2 3 f32Payload
    use weights = device.UploadWeights(matrix)
    let ids = [| 1; 0 |]
    let expected = cpuLookup matrix ids
    use destination = device.Allocate(TensorShape.Create [| 2; 3 |])
    device.EmbeddingLookup(weights, ReadOnlyMemory<int>(ids), destination)
    // The caller may reuse its ID buffer immediately after the call returns.
    Array.fill ids 0 ids.Length 999
    // Queued after the lookup on the same stream: destination doubles.
    device.Add(destination, destination, destination)
    Assert.Equal<single>(Array.map (fun value -> value + value) expected, download device destination)

[<CudaFact>]
[<Trait("Category", "CUDA")>]
let ``CUDA embedding lookup rejects bad shapes IDs encodings and ownership before writing`` () =
    use backend = new CudaBackend()
    use other = new CudaBackend()
    let device = backend :> ITensorBackend
    let matrix = matrixOf TensorEncoding.F32 2 3 f32Payload
    use weights = device.UploadWeights(matrix)
    use foreignWeights = (other :> ITensorBackend).UploadWeights(matrix)
    let sentinel = Array.create 6 42.0f
    use destination = device.Upload(TensorShape.Create [| 2; 3 |], ReadOnlyMemory<single>(sentinel))
    use foreignDestination = (other :> ITensorBackend).Upload(TensorShape.Create [| 2; 3 |], ReadOnlyMemory<single>(sentinel))
    use halfDestination = device.Allocate(TensorShape.Create [| 2; 3 |], TensorEncoding.F16)
    use wrongShape = device.Allocate(TensorShape.Create [| 3; 2 |])
    use empty = device.Allocate(TensorShape.Create [| 0; 3 |])
    let ids = ReadOnlyMemory<int>([| 1; 0 |])

    Assert.Throws<ArgumentNullException>(fun () ->
        device.EmbeddingLookup(Unchecked.defaultof<IWeightTensor>, ids, destination)) |> ignore
    Assert.Throws<ArgumentNullException>(fun () ->
        device.EmbeddingLookup(weights, ids, Unchecked.defaultof<ITensor>)) |> ignore
    Assert.Throws<ArgumentException>(fun () -> device.EmbeddingLookup(foreignWeights, ids, destination)) |> ignore
    Assert.Throws<ArgumentException>(fun () -> device.EmbeddingLookup(weights, ids, foreignDestination)) |> ignore
    Assert.Throws<ArgumentException>(fun () -> device.EmbeddingLookup(weights, ids, halfDestination)) |> ignore
    Assert.Throws<ArgumentException>(fun () -> device.EmbeddingLookup(weights, ids, wrongShape)) |> ignore

    // An empty batch is a validated no-op.
    device.EmbeddingLookup(weights, ReadOnlyMemory<int>.Empty, empty)
    Assert.Empty(download device empty)

    // The whole batch is validated first: a bad final ID leaves every row unwritten.
    for invalid in [ -1; 2; Int32.MinValue; Int32.MaxValue ] do
        let error = Assert.Throws<ArgumentOutOfRangeException>(fun () ->
            device.EmbeddingLookup(weights, ReadOnlyMemory<int>([| 1; invalid |]), destination))
        Assert.Equal("tokenIds", error.ParamName)
        Assert.Contains("index 1", error.Message)
    Assert.Equal<single>(sentinel, download device destination)

    // Unsupported stored encodings are rejected before submission, never decoded another way.
    for encoding in [ TensorEncoding.BF16; TensorEncoding.Q4_1; TensorEncoding.Q5_0; TensorEncoding.Q5_1
                      TensorEncoding.IQ4_NL; TensorEncoding.Q2_K; TensorEncoding.Q3_K; TensorEncoding.Q4_K
                      TensorEncoding.Q5_K; TensorEncoding.Q6_K ] do
        let columns = encoding.BlockElements
        let unsupported = matrixOf encoding 2 columns (Array.zeroCreate (2 * encoding.BlockBytes))
        use unsupportedWeights = device.UploadWeights(unsupported)
        use unsupportedDestination = device.Allocate(TensorShape.Create [| 2; columns |])
        Assert.Throws<NotSupportedException>(fun () ->
            device.EmbeddingLookup(unsupportedWeights, ids, unsupportedDestination)) |> ignore
        Assert.Equal<single>(Array.zeroCreate (2 * columns), download device unsupportedDestination)
    Assert.Equal<single>(sentinel, download device destination)

    use disposedWeights = device.UploadWeights(matrix)
    disposedWeights.Dispose()
    Assert.Throws<ObjectDisposedException>(fun () -> device.EmbeddingLookup(disposedWeights, ids, destination)) |> ignore
    use disposedDestination = device.Allocate(TensorShape.Create [| 2; 3 |])
    disposedDestination.Dispose()
    Assert.Throws<ObjectDisposedException>(fun () -> device.EmbeddingLookup(weights, ids, disposedDestination)) |> ignore
    Assert.Equal<single>(sentinel, download device destination)

module private Native =
    [<DllImport("marmotte_cuda", CallingConvention = CallingConvention.Cdecl)>]
    extern int marmotte_cuda_create(int device, nativeint& context)
    [<DllImport("marmotte_cuda", CallingConvention = CallingConvention.Cdecl)>]
    extern int marmotte_cuda_destroy(nativeint context)
    [<DllImport("marmotte_cuda", CallingConvention = CallingConvention.Cdecl)>]
    extern int marmotte_cuda_allocate(nativeint context, int count, nativeint& buffer)
    [<DllImport("marmotte_cuda", CallingConvention = CallingConvention.Cdecl)>]
    extern int marmotte_cuda_allocate_f16(nativeint context, int count, nativeint& buffer)
    [<DllImport("marmotte_cuda", CallingConvention = CallingConvention.Cdecl)>]
    extern int marmotte_cuda_allocate_stored(nativeint context, int encoding, int64 bytes, nativeint& buffer)
    [<DllImport("marmotte_cuda", CallingConvention = CallingConvention.Cdecl)>]
    extern int marmotte_cuda_upload(nativeint context, nativeint buffer, single[] values, int count)
    [<DllImport("marmotte_cuda", CallingConvention = CallingConvention.Cdecl)>]
    extern int marmotte_cuda_download(nativeint context, nativeint buffer, [<Out>] single[] values, int count)
    [<DllImport("marmotte_cuda", CallingConvention = CallingConvention.Cdecl)>]
    extern int marmotte_cuda_upload_stored_range(nativeint context, nativeint buffer, int64 offset, byte[] payload, int64 bytes)
    [<DllImport("marmotte_cuda", CallingConvention = CallingConvention.Cdecl)>]
    extern int marmotte_cuda_embedding(nativeint context, nativeint weights, int[] tokenIds, nativeint destination, int tokens, int rows, int columns, int rowBytes)
    [<DllImport("marmotte_cuda", CallingConvention = CallingConvention.Cdecl)>]
    extern int marmotte_cuda_free(nativeint context, nativeint buffer)

[<CudaFact>]
[<Trait("Category", "CUDA")>]
let ``CUDA native embedding rejects inconsistent counts encodings and foreign pointers`` () =
    let mutable context = 0n
    let mutable other = 0n
    Assert.Equal(0, Native.marmotte_cuda_create(0, &context))
    try
        Assert.Equal(0, Native.marmotte_cuda_create(0, &other))
        let mutable weights = 0n
        let mutable unsupported = 0n
        let mutable destination = 0n
        let mutable half = 0n
        // F32 weights: 2 rows x 3 columns = 24 stored bytes, rowBytes = 12.
        Assert.Equal(0, Native.marmotte_cuda_allocate_stored(context, 0, 24L, &weights))
        Assert.Equal(0, Native.marmotte_cuda_allocate_stored(context, 13, 420L, &unsupported)) // Q6_K, 2 x 210
        Assert.Equal(0, Native.marmotte_cuda_allocate(context, 6, &destination))
        Assert.Equal(0, Native.marmotte_cuda_allocate_f16(context, 6, &half))
        Assert.Equal(0, Native.marmotte_cuda_upload(context, destination, Array.create 6 42.0f, 6))
        Assert.Equal(0, Native.marmotte_cuda_upload_stored_range(context, weights, 0L, Array.init 24 (fun i -> byte (i + 1)), 24L))
        let ids = [| 1; 0 |]

        // Every invalid call must fail before any destination write.
        let invalidCalls =
            [ context, weights, ids, destination, 0, 2, 3, 12
              context, weights, ids, destination, -1, 2, 3, 12
              context, weights, ids, destination, 2, 0, 3, 12
              context, weights, ids, destination, 2, 2, 0, 12
              context, weights, ids, destination, 2, 2, 3, 0
              context, weights, ids, destination, 2, 2, 3, 11
              context, weights, ids, destination, 2, 2, 4, 16
              context, weights, ids, destination, 2, 3, 3, 12
              context, weights, ids, destination, Int32.MaxValue, 2, 2, 8
              context, weights, ids, half, 2, 2, 3, 12
              context, weights, ids, weights, 2, 2, 3, 12
              context, destination, ids, destination, 2, 2, 3, 12
              context, unsupported, ids, destination, 2, 2, 256, 210
              other, weights, ids, destination, 2, 2, 3, 12 ]
        for callContext, stored, batch, into, tokens, rows, columns, rowBytes in invalidCalls do
            Assert.NotEqual(0, Native.marmotte_cuda_embedding(callContext, stored, batch, into, tokens, rows, columns, rowBytes))

        let actual = Array.create 6 0.0f
        Assert.Equal(0, Native.marmotte_cuda_download(context, destination, actual, 6))
        Assert.All(actual, fun value -> Assert.Equal(42.0f, value))

        // The valid call decodes row 1 then row 0 of the uploaded F32 payload (bytes 1..24,
        // little-endian floats): row 0 starts at 0x04030201, row 1 at 0x100F0E0D.
        Assert.Equal(0, Native.marmotte_cuda_embedding(context, weights, ids, destination, 2, 2, 3, 12))
        Assert.Equal(0, Native.marmotte_cuda_download(context, destination, actual, 6))
        let row0 = [| 0x04030201; 0x08070605; 0x0C0B0A09 |] |> Array.map BitConverter.Int32BitsToSingle
        let row1 = [| 0x100F0E0D; 0x14131211; 0x18171615 |] |> Array.map BitConverter.Int32BitsToSingle
        Assert.Equal<single>(Array.append row1 row0, actual)

        Assert.Equal(0, Native.marmotte_cuda_free(context, weights))
        Assert.Equal(0, Native.marmotte_cuda_free(context, unsupported))
        Assert.Equal(0, Native.marmotte_cuda_free(context, destination))
        Assert.Equal(0, Native.marmotte_cuda_free(context, half))
    finally
        if other <> 0n then Assert.Equal(0, Native.marmotte_cuda_destroy(other))
        Assert.Equal(0, Native.marmotte_cuda_destroy(context))
