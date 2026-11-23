module Marmotte.Backends.CudaMatMulTests

open System
open System.Buffers.Binary
open System.Collections.Generic
open System.Runtime.InteropServices
open Marmotte.Core
open Marmotte.Cpu
open Marmotte.Cuda
open Marmotte.Backends.CudaTests
open Xunit

/// Serves one matrix from memory, to build a WeightMatrix without a file format.
type private MatrixSource(encoding: TensorEncoding, rows: int, columns: int, payload: byte array) =
    let descriptor =
        { Name = "blk.0.attn_q.weight"
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
    use source = new MatrixSource(encoding, rows, columns, payload)
    WeightMatrix.Read(source, "blk.0.attn_q.weight")

let private download (backend: ITensorBackend) (tensor: ITensor) =
    let values = Array.zeroCreate<single> tensor.Shape.ElementCount
    backend.Download(tensor, values.AsMemory())
    values

/// Random F32/F16 weights in [-1, 1), packed little-endian; the matrix product fixtures only
/// cover the two encodings CUDA implements.
let private randomPayload (random: Random) (encoding: TensorEncoding) rows columns =
    let valueCount = rows * columns
    let floats = Array.init valueCount (fun _ -> float32 (random.NextDouble() * 2.0 - 1.0))
    match encoding with
    | TensorEncoding.F32 ->
        let bytes = Array.zeroCreate<byte> (4 * valueCount)
        floats |> Array.iteri (fun index value ->
            BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(4 * index, 4), value))
        bytes
    | TensorEncoding.F16 ->
        let bytes = Array.zeroCreate<byte> (2 * valueCount)
        floats |> Array.iteri (fun index value ->
            BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(2 * index, 2), BitConverter.HalfToUInt16Bits(Half.op_Explicit value)))
        bytes
    | _ -> invalidArg (nameof encoding) "The fixtures cover F32 and F16 weights only."

/// Two F32 rows [1; 2; 3] and [4; 5; 6], little-endian.
let private f32Payload =
    [| 1.0f; 2.0f; 3.0f; 4.0f; 5.0f; 6.0f |]
    |> Array.collect BitConverter.GetBytes

/// Checks MatVec against decoded rows multiplied in double precision, with a tolerance relative
/// to the sum of absolute products, since cancellation can make an output much smaller than its terms.
let private assertMatVec (backend: ITensorBackend) (matrix: WeightMatrix) (inputValues: single array) =
    use weights = backend.UploadWeights(matrix)
    use input = backend.Upload(TensorShape.Create [| matrix.Columns |], ReadOnlyMemory<single>(inputValues))
    use destination = backend.Allocate(TensorShape.Create [| matrix.Rows |])

    backend.MatVec(weights, input, destination)

    let result = download backend destination
    let decoded = Array.zeroCreate<single> matrix.Columns
    for row in 0 .. matrix.Rows - 1 do
        matrix.DecodeRow(row, Span<single>(decoded))
        let products = Array.map2 (fun (weight: single) (value: single) -> float weight * float value) decoded inputValues
        let expected = Array.sum products
        let magnitude = products |> Array.sumBy abs
        let difference = abs (float result[row] - expected)
        Assert.True(difference <= 1e-6 * magnitude + 1e-30, $"Row {row}: expected {expected}, got {result[row]}.")
    Assert.Equal<single>(inputValues, download backend input)

/// Checks MatMul token by token against decoded rows multiplied in double precision, with the
/// same tolerance as MatVec.
let private assertMatMul (backend: ITensorBackend) (matrix: WeightMatrix) tokens (inputValues: single array) =
    use weights = backend.UploadWeights(matrix)
    use input = backend.Upload(TensorShape.Create [| tokens; matrix.Columns |], ReadOnlyMemory<single>(inputValues))
    use destination = backend.Allocate(TensorShape.Create [| tokens; matrix.Rows |])

    backend.MatMul(weights, input, destination)

    let result = download backend destination
    let decoded = Array.zeroCreate<single> matrix.Columns
    for row in 0 .. matrix.Rows - 1 do
        matrix.DecodeRow(row, Span<single>(decoded))
        for token in 0 .. tokens - 1 do
            let tokenInput = inputValues[token * matrix.Columns .. (token + 1) * matrix.Columns - 1]
            let products = Array.map2 (fun (weight: single) (value: single) -> float weight * float value) decoded tokenInput
            let expected = Array.sum products
            let actual = result[token * matrix.Rows + row]
            let tolerance = 1e-6 * (products |> Array.sumBy abs) + 1e-30
            Assert.True(abs (float actual - expected) <= tolerance, $"Token {token}, row {row}: expected {expected}, got {actual}.")
    Assert.Equal<single>(inputValues, download backend input)

[<Fact>]
let ``CUDA matrix products validate null and disposed arguments without a driver`` () =
    use backend = new CudaBackend() :> ITensorBackend
    let nullWeights = Unchecked.defaultof<IWeightTensor>
    let nullTensor = Unchecked.defaultof<ITensor>
    Assert.Throws<ArgumentNullException>(fun () -> backend.MatVec(nullWeights, nullTensor, nullTensor)) |> ignore
    Assert.Throws<ArgumentNullException>(fun () -> backend.MatMul(nullWeights, nullTensor, nullTensor)) |> ignore
    backend.Dispose()
    Assert.Throws<ObjectDisposedException>(fun () -> backend.MatVec(nullWeights, nullTensor, nullTensor)) |> ignore
    Assert.Throws<ObjectDisposedException>(fun () -> backend.MatMul(nullWeights, nullTensor, nullTensor)) |> ignore

[<CudaFact>]
[<Trait("Category", "CUDA")>]
let ``CUDA MatVec matches decoded rows in double precision for F32 and F16`` () =
    use backend = new CudaBackend() :> ITensorBackend
    let random = Random(2718)
    // Widths cover the scalar tail, one block-worth and several chunks of the batched kernel.
    let cases =
        [ TensorEncoding.F32, [ 1; 3; 32; 67; 256; 1024; 3072 ]
          TensorEncoding.F16, [ 1; 5; 64; 257; 1024; 3072 ] ]

    for encoding, widths in cases do
        for columns in widths do
            let matrix = matrixOf encoding 7 columns (randomPayload random encoding 7 columns)
            let inputValues = Array.init columns (fun _ -> float32 (random.NextDouble() * 2.0 - 1.0))
            assertMatVec backend matrix inputValues

    // One larger matrix, closer to real layer shapes.
    let matrix = matrixOf TensorEncoding.F32 257 3072 (randomPayload random TensorEncoding.F32 257 3072)
    let inputValues = Array.init 3072 (fun _ -> float32 (random.NextDouble() * 2.0 - 1.0))
    assertMatVec backend matrix inputValues

[<CudaFact>]
[<Trait("Category", "CUDA")>]
let ``CUDA MatMul matches decoded rows in double precision for F32 F16 and batch sizes`` () =
    use backend = new CudaBackend() :> ITensorBackend
    let random = Random(3141)
    for encoding, columns in [ TensorEncoding.F32, 67; TensorEncoding.F16, 64; TensorEncoding.F32, 3073 ] do
        let matrix = matrixOf encoding 7 columns (randomPayload random encoding 7 columns)
        // 300 tokens exercises the native slicing of batches larger than one block.
        for tokens in [ 1; 2; 5; 64; 300 ] do
            let inputValues = Array.init (tokens * columns) (fun _ -> float32 (random.NextDouble() * 2.0 - 1.0))
            assertMatMul backend matrix tokens inputValues

[<CudaFact>]
[<Trait("Category", "CUDA")>]
let ``CUDA MatMul gives each token its MatVec result and matches the CPU backend`` () =
    use backend = new CudaBackend() :> ITensorBackend
    use cpu = new CpuBackend(CpuMatrixProductMode.DecodedFloat) :> ITensorBackend
    let random = Random(4242)
    let matrix = matrixOf TensorEncoding.F16 65 512 (randomPayload random TensorEncoding.F16 65 512)
    let inputValues = Array.init (4 * 512) (fun _ -> float32 (random.NextDouble() * 2.0 - 1.0))

    use weights = backend.UploadWeights(matrix)
    use cpuWeights = cpu.UploadWeights(matrix)
    use input = backend.Upload(TensorShape.Create [| 4; 512 |], ReadOnlyMemory<single>(inputValues))
    use destination = backend.Allocate(TensorShape.Create [| 4; 65 |])
    backend.MatMul(weights, input, destination)
    let batched = download backend destination

    use cpuInput = cpu.Upload(TensorShape.Create [| 4; 512 |], ReadOnlyMemory<single>(inputValues))
    use cpuDestination = cpu.Allocate(TensorShape.Create [| 4; 65 |])
    cpu.MatMul(cpuWeights, cpuInput, cpuDestination)
    let cpuResult = download cpu cpuDestination
    for index in 0 .. batched.Length - 1 do
        let magnitude = max (abs (float batched[index])) (abs (float cpuResult[index]))
        Assert.True(abs (float batched[index] - float cpuResult[index]) <= 1e-6 * magnitude + 1e-30,
                    $"Index {index}: CUDA {batched[index]}, CPU {cpuResult[index]}.")

    for token in 0 .. 3 do
        use vector = backend.Upload(TensorShape.Create [| 512 |], ReadOnlyMemory<single>(inputValues, token * 512, 512))
        use output = backend.Allocate(TensorShape.Create [| 65 |])
        backend.MatVec(weights, vector, output)
        let single = download backend output
        for row in 0 .. 64 do
            let magnitude = max (abs (float single[row])) (abs (float batched[token * 65 + row]))
            Assert.True(abs (float single[row] - float batched[token * 65 + row]) <= 1e-6 * magnitude + 1e-30,
                        $"Token {token}, row {row}: MatVec {single[row]}, MatMul {batched[token * 65 + row]}.")

[<CudaFact>]
[<Trait("Category", "CUDA")>]
let ``CUDA MatMul accepts an empty batch as a no-op and leaves stored weights unchanged`` () =
    use backend = new CudaBackend()
    let device = backend :> ITensorBackend
    let matrix = matrixOf TensorEncoding.F32 2 3 f32Payload
    let original = Array.copy f32Payload
    use weights = device.UploadWeights(matrix)
    use input = device.Allocate(TensorShape.Create [| 0; 3 |])
    use destination = device.Allocate(TensorShape.Create [| 0; 2 |])

    device.MatMul(weights, input, destination)
    device.Synchronize()

    Assert.Empty(download device destination)
    Assert.Equal<byte>(original, backend.DownloadWeights(weights))

[<CudaFact>]
[<Trait("Category", "CUDA")>]
let ``CUDA matrix products validate shapes encodings ownership and disposal before writing`` () =
    use backend = new CudaBackend() :> ITensorBackend
    use otherBackend = new CudaBackend() :> ITensorBackend
    use cpu = new CpuBackend() :> ITensorBackend
    let matrix = matrixOf TensorEncoding.F32 4 4 (Array.zeroCreate 64)
    use weights = backend.UploadWeights(matrix)
    use foreignWeights = otherBackend.UploadWeights(matrix)
    let disposedWeights = backend.UploadWeights(matrix)
    disposedWeights.Dispose()
    let vector () = backend.Upload(TensorShape.Create [| 4 |], ReadOnlyMemory<single>(Array.create 4 1.0f))
    use input = vector ()
    use destination = backend.Upload(TensorShape.Create [| 4 |], ReadOnlyMemory<single>(Array.create 4 42.0f))
    use shortInput = backend.Allocate(TensorShape.Create [| 3 |])
    use longDestination = backend.Allocate(TensorShape.Create [| 5 |])
    use matrixInput = backend.Allocate(TensorShape.Create [| 1; 4 |])
    use halfInput = backend.Allocate(TensorShape.Create [| 4 |], TensorEncoding.F16)
    use foreignInput = otherBackend.Allocate(TensorShape.Create [| 4 |])
    use hostInput = cpu.Allocate(TensorShape.Create [| 4 |])
    let disposedInput = vector ()
    disposedInput.Dispose()

    let shapeError = Assert.Throws<ArgumentException>(fun () -> backend.MatVec(weights, shortInput, destination))
    Assert.Contains("[4]", shapeError.Message)
    Assert.Contains("[3]", shapeError.Message)
    Assert.Throws<ArgumentException>(fun () -> backend.MatVec(weights, input, longDestination)) |> ignore
    Assert.Throws<ArgumentException>(fun () -> backend.MatVec(weights, matrixInput, destination)) |> ignore
    Assert.Throws<ArgumentException>(fun () -> backend.MatVec(weights, halfInput, destination)) |> ignore
    Assert.Throws<ArgumentException>(fun () -> backend.MatVec(weights, input, input)) |> ignore
    Assert.Throws<ArgumentException>(fun () -> backend.MatVec(foreignWeights, input, destination)) |> ignore
    Assert.Throws<ArgumentException>(fun () -> backend.MatVec(weights, foreignInput, destination)) |> ignore
    Assert.Throws<ArgumentException>(fun () -> backend.MatVec(weights, hostInput, destination)) |> ignore
    Assert.Throws<ObjectDisposedException>(fun () -> backend.MatVec(disposedWeights, input, destination)) |> ignore
    Assert.Throws<ObjectDisposedException>(fun () -> backend.MatVec(weights, disposedInput, destination)) |> ignore

    // MatMul takes [tokens; columns] inputs and [tokens; rows] destinations.
    use batch = backend.Upload(TensorShape.Create [| 2; 4 |], ReadOnlyMemory<single>(Array.create 8 1.0f))
    use batchDestination = backend.Upload(TensorShape.Create [| 2; 4 |], ReadOnlyMemory<single>(Array.create 8 42.0f))
    use narrowBatch = backend.Allocate(TensorShape.Create [| 2; 3 |])
    use otherTokens = backend.Allocate(TensorShape.Create [| 3; 4 |])
    use wideBatchDestination = backend.Allocate(TensorShape.Create [| 2; 5 |])
    use halfBatch = backend.Allocate(TensorShape.Create [| 2; 4 |], TensorEncoding.F16)
    let batchError = Assert.Throws<ArgumentException>(fun () -> backend.MatMul(weights, narrowBatch, batchDestination))
    Assert.Contains("[tokens; 4]", batchError.Message)
    Assert.Contains("[2; 3]", batchError.Message)
    Assert.Throws<ArgumentException>(fun () -> backend.MatMul(weights, input, batchDestination)) |> ignore
    Assert.Throws<ArgumentException>(fun () -> backend.MatMul(weights, batch, otherTokens)) |> ignore
    Assert.Throws<ArgumentException>(fun () -> backend.MatMul(weights, batch, wideBatchDestination)) |> ignore
    Assert.Throws<ArgumentException>(fun () -> backend.MatMul(weights, batch, batch)) |> ignore
    Assert.Throws<ArgumentException>(fun () -> backend.MatMul(weights, halfBatch, batchDestination)) |> ignore
    Assert.Throws<ArgumentException>(fun () -> backend.MatMul(foreignWeights, batch, batchDestination)) |> ignore
    Assert.Throws<ArgumentException>(fun () -> backend.MatMul(weights, foreignInput, batchDestination)) |> ignore
    Assert.Throws<ObjectDisposedException>(fun () -> backend.MatMul(disposedWeights, batch, batchDestination)) |> ignore

    // Unsupported stored encodings are rejected before submission, never decoded another way.
    for encoding in [ TensorEncoding.BF16; TensorEncoding.Q4_0; TensorEncoding.Q4_1; TensorEncoding.Q5_0
                      TensorEncoding.Q5_1; TensorEncoding.Q8_0; TensorEncoding.IQ4_NL; TensorEncoding.Q2_K
                      TensorEncoding.Q3_K; TensorEncoding.Q4_K; TensorEncoding.Q5_K; TensorEncoding.Q6_K ] do
        let columns = encoding.BlockElements
        let unsupported = matrixOf encoding 2 columns (Array.zeroCreate (2 * encoding.BlockBytes))
        use unsupportedWeights = backend.UploadWeights(unsupported)
        use vectorInput = backend.Allocate(TensorShape.Create [| columns |])
        use vectorDestination = backend.Allocate(TensorShape.Create [| 2 |])
        use batchInput = backend.Allocate(TensorShape.Create [| 3; columns |])
        use batchOutput = backend.Allocate(TensorShape.Create [| 3; 2 |])
        Assert.Throws<NotSupportedException>(fun () -> backend.MatVec(unsupportedWeights, vectorInput, vectorDestination)) |> ignore
        Assert.Throws<NotSupportedException>(fun () -> backend.MatMul(unsupportedWeights, batchInput, batchOutput)) |> ignore
        Assert.Equal<single>(Array.zeroCreate columns, download backend vectorInput)
        Assert.Equal<single>(Array.zeroCreate 2, download backend vectorDestination)
        Assert.Equal<single>(Array.zeroCreate (3 * 2), download backend batchOutput)

    Assert.Equal<single>(Array.create 4 42.0f, download backend destination)
    Assert.Equal<single>(Array.create 8 42.0f, download backend batchDestination)

[<CudaFact>]
[<Trait("Category", "CUDA")>]
let ``CUDA matrix products run in stream order between dependent operations`` () =
    use backend = new CudaBackend() :> ITensorBackend
    use cpu = new CpuBackend(CpuMatrixProductMode.DecodedFloat) :> ITensorBackend
    let matrix = matrixOf TensorEncoding.F32 2 3 f32Payload
    use weights = backend.UploadWeights(matrix)
    use cpuWeights = cpu.UploadWeights(matrix)
    use input = backend.Upload(TensorShape.Create [| 3 |], ReadOnlyMemory<single>([| 1.0f; 1.0f; 1.0f |]))
    use destination = backend.Upload(TensorShape.Create [| 2 |], ReadOnlyMemory<single>([| 42.0f; 42.0f |]))

    // The queued Add doubles the input before the MatVec reads it; the second Add doubles
    // the MatVec result afterwards. Expected: 2 * [1+2+3; 4+5+6] = [24; 60].
    backend.Add(input, input, input)
    backend.MatVec(weights, input, destination)
    backend.Add(destination, destination, destination)

    use cpuInput = cpu.Upload(TensorShape.Create [| 3 |], ReadOnlyMemory<single>([| 1.0f; 1.0f; 1.0f |]))
    use cpuDestination = cpu.Upload(TensorShape.Create [| 2 |], ReadOnlyMemory<single>([| 42.0f; 42.0f |]))
    cpu.Add(cpuInput, cpuInput, cpuInput)
    cpu.MatVec(cpuWeights, cpuInput, cpuDestination)
    cpu.Add(cpuDestination, cpuDestination, cpuDestination)

    Assert.Equal<single>(download cpu cpuDestination, download backend destination)
    Assert.Equal<single>([| 24.0f; 60.0f |], download backend destination)
    Assert.All(download backend input, fun value -> Assert.Equal(2.0f, value))

    // A batched product observes an earlier queued Add on its input, and a later Add observes it.
    use batchInput = backend.Upload(TensorShape.Create [| 2; 3 |], ReadOnlyMemory<single>([| 1.0f; 2.0f; 3.0f; 0.5f; -1.0f; 2.0f |]))
    use batchDestination = backend.Allocate(TensorShape.Create [| 2; 2 |])
    backend.Add(batchInput, batchInput, batchInput)
    backend.MatMul(weights, batchInput, batchDestination)
    backend.Add(batchDestination, batchDestination, batchDestination)

    use cpuBatchInput = cpu.Upload(TensorShape.Create [| 2; 3 |], ReadOnlyMemory<single>([| 1.0f; 2.0f; 3.0f; 0.5f; -1.0f; 2.0f |]))
    use cpuBatchDestination = cpu.Allocate(TensorShape.Create [| 2; 2 |])
    cpu.Add(cpuBatchInput, cpuBatchInput, cpuBatchInput)
    cpu.MatMul(cpuWeights, cpuBatchInput, cpuBatchDestination)
    cpu.Add(cpuBatchDestination, cpuBatchDestination, cpuBatchDestination)

    Assert.Equal<single>(download cpu cpuBatchDestination, download backend batchDestination)
    // Stored weights are never written.
    Assert.Equal<byte>(f32Payload, (backend :?> CudaBackend).DownloadWeights(weights))

[<CudaFact>]
[<Trait("Category", "CUDA")>]
let ``CUDA matrix products propagate NaN and infinity from F32 and F16 weights`` () =
    use backend = new CudaBackend() :> ITensorBackend
    // Rows: [NaN; 1; 1], [+Inf; 0; 0], [1; -Inf; 1].
    let special =
        [| Single.NaN; 1.0f; 1.0f
           Single.PositiveInfinity; 0.0f; 0.0f
           1.0f; Single.NegativeInfinity; 1.0f |]
        |> Array.collect BitConverter.GetBytes
    let matrix = matrixOf TensorEncoding.F32 3 3 special
    use weights = backend.UploadWeights(matrix)
    use input = backend.Upload(TensorShape.Create [| 3 |], ReadOnlyMemory<single>([| 1.0f; 2.0f; 3.0f |]))
    use destination = backend.Allocate(TensorShape.Create [| 3 |])
    backend.MatVec(weights, input, destination)
    let result = download backend destination
    Assert.True(Single.IsNaN result[0])
    Assert.Equal(Single.PositiveInfinity, result[1])
    Assert.Equal(Single.NegativeInfinity, result[2])

    // F16 rows [NaN; 1] and [+Inf; 0], as raw little-endian bits.
    let halfBits = [| 0x7E01us; 0x3C00us; 0x7C00us; 0x0000us |]
    let halfPayload = halfBits |> Array.collect (fun bits -> [| byte bits; byte (bits >>> 8) |])
    let halfMatrix = matrixOf TensorEncoding.F16 2 2 halfPayload
    use halfWeights = backend.UploadWeights(halfMatrix)
    use halfInput = backend.Upload(TensorShape.Create [| 2 |], ReadOnlyMemory<single>([| 1.0f; 1.0f |]))
    use halfDestination = backend.Allocate(TensorShape.Create [| 2 |])
    backend.MatVec(halfWeights, halfInput, halfDestination)
    let halfResult = download backend halfDestination
    Assert.True(Single.IsNaN halfResult[0])
    Assert.Equal(Single.PositiveInfinity, halfResult[1])

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
    extern int marmotte_cuda_matvec(nativeint context, nativeint weights, nativeint input, nativeint destination, int rows, int columns, int rowBytes)
    [<DllImport("marmotte_cuda", CallingConvention = CallingConvention.Cdecl)>]
    extern int marmotte_cuda_matmul(nativeint context, nativeint weights, nativeint input, nativeint destination, int tokens, int rows, int columns, int rowBytes)
    [<DllImport("marmotte_cuda", CallingConvention = CallingConvention.Cdecl)>]
    extern int marmotte_cuda_free(nativeint context, nativeint buffer)

[<CudaFact>]
[<Trait("Category", "CUDA")>]
let ``CUDA native matrix products reject inconsistent counts encodings aliasing and foreign pointers`` () =
    let mutable context = 0n
    let mutable other = 0n
    Assert.Equal(0, Native.marmotte_cuda_create(0, &context))
    try
        Assert.Equal(0, Native.marmotte_cuda_create(0, &other))
        let mutable weights = 0n
        let mutable quantized = 0n
        let mutable input = 0n
        let mutable shortDestination = 0n
        let mutable batchDestination = 0n
        let mutable half = 0n
        // F32 weights: 2 rows x 3 columns = 24 stored bytes, rowBytes = 12.
        let mutable vectorInput = 0n
        Assert.Equal(0, Native.marmotte_cuda_allocate_stored(context, 0, 24L, &weights))
        // Q4_0: 2 rows x 32 columns, rowBytes = 18, an encoding the products reject.
        Assert.Equal(0, Native.marmotte_cuda_allocate_stored(context, 3, 36L, &quantized))
        Assert.Equal(0, Native.marmotte_cuda_allocate(context, 3, &vectorInput))
        Assert.Equal(0, Native.marmotte_cuda_allocate(context, 6, &input))
        Assert.Equal(0, Native.marmotte_cuda_allocate(context, 2, &shortDestination))
        Assert.Equal(0, Native.marmotte_cuda_allocate(context, 4, &batchDestination))
        Assert.Equal(0, Native.marmotte_cuda_allocate_f16(context, 6, &half))
        Assert.Equal(0, Native.marmotte_cuda_upload_stored_range(context, weights, 0L, f32Payload, 24L))
        Assert.Equal(0, Native.marmotte_cuda_upload(context, vectorInput, [| 2.0f; 1.0f; -1.0f |], 3))
        Assert.Equal(0, Native.marmotte_cuda_upload(context, input, [| 2.0f; 1.0f; -1.0f; 1.0f; 1.0f; 1.0f |], 6))
        Assert.Equal(0, Native.marmotte_cuda_upload(context, shortDestination, [| 42.0f; 42.0f |], 2))
        Assert.Equal(0, Native.marmotte_cuda_upload(context, batchDestination, [| 42.0f; 42.0f; 42.0f; 42.0f |], 4))

        // Every invalid call must fail before any destination write.
        let invalidMatVec =
            [ context, weights, vectorInput, shortDestination, 0, 3, 12
              context, weights, vectorInput, shortDestination, 2, 0, 12
              context, weights, vectorInput, shortDestination, 2, 3, 0
              context, weights, vectorInput, shortDestination, 2, 3, 11
              context, weights, vectorInput, shortDestination, 2, 3, 16
              context, weights, vectorInput, shortDestination, 2, 4, 16 // input would need 4 elements, not 3
              context, weights, input, shortDestination, 2, 3, 12      // input would need 3 elements, not 6
              context, weights, vectorInput, shortDestination, 3, 3, 12 // destination would need 3, not 2
              context, weights, vectorInput, vectorInput, 2, 3, 12      // input == destination
              context, vectorInput, vectorInput, shortDestination, 2, 3, 12 // execution buffer as weights
              context, weights, half, shortDestination, 2, 3, 12     // F16 input
              context, weights, vectorInput, half, 2, 3, 12          // F16 destination
              context, quantized, vectorInput, shortDestination, 2, 32, 18 // unsupported encoding
              other, weights, vectorInput, shortDestination, 2, 3, 12 ]
        for callContext, stored, into, out, rows, columns, rowBytes in invalidMatVec do
            Assert.NotEqual(0, Native.marmotte_cuda_matvec(callContext, stored, into, out, rows, columns, rowBytes))

        let invalidMatMul =
            [ context, weights, input, batchDestination, 0, 2, 3, 12
              context, weights, input, batchDestination, -1, 2, 3, 12
              context, weights, input, batchDestination, 3, 2, 3, 12  // input would need 9 elements, not 6
              context, weights, input, batchDestination, 2, 3, 3, 12  // weights would need 3 rows, not 2
              context, weights, input, input, 2, 2, 3, 12             // input == destination
              context, weights, input, half, 2, 2, 3, 12              // F16 destination
              context, quantized, input, batchDestination, 2, 2, 32, 18
              other, weights, input, batchDestination, 2, 2, 3, 12 ]
        for callContext, stored, into, out, tokens, rows, columns, rowBytes in invalidMatMul do
            Assert.NotEqual(0, Native.marmotte_cuda_matmul(callContext, stored, into, out, tokens, rows, columns, rowBytes))

        let shortResult = Array.create 2 0.0f
        let batchResult = Array.create 4 0.0f
        Assert.Equal(0, Native.marmotte_cuda_download(context, shortDestination, shortResult, 2))
        Assert.Equal(0, Native.marmotte_cuda_download(context, batchDestination, batchResult, 4))
        Assert.All(shortResult, fun value -> Assert.Equal(42.0f, value))
        Assert.All(batchResult, fun value -> Assert.Equal(42.0f, value))

        // MatVec with input [2; 1; -1]: row 0 gives 1*2 + 2*1 + 3*(-1) = 1, row 1 gives 8 + 5 - 6 = 7.
        Assert.Equal(0, Native.marmotte_cuda_matvec(context, weights, vectorInput, shortDestination, 2, 3, 12))
        Assert.Equal(0, Native.marmotte_cuda_download(context, shortDestination, shortResult, 2))
        Assert.Equal<single>([| 1.0f; 7.0f |], shortResult)

        // MatMul with two tokens: token 0 is [2; 1; -1] as above, token 1 is [1; 1; 1], giving
        // rows [6; 15]. The destination is row-major [token; row].
        Assert.Equal(0, Native.marmotte_cuda_matmul(context, weights, input, batchDestination, 2, 2, 3, 12))
        Assert.Equal(0, Native.marmotte_cuda_download(context, batchDestination, batchResult, 4))
        Assert.Equal<single>([| 1.0f; 7.0f; 6.0f; 15.0f |], batchResult)

        Assert.Equal(0, Native.marmotte_cuda_free(context, weights))
        Assert.Equal(0, Native.marmotte_cuda_free(context, quantized))
        Assert.Equal(0, Native.marmotte_cuda_free(context, vectorInput))
        Assert.Equal(0, Native.marmotte_cuda_free(context, input))
        Assert.Equal(0, Native.marmotte_cuda_free(context, shortDestination))
        Assert.Equal(0, Native.marmotte_cuda_free(context, batchDestination))
        Assert.Equal(0, Native.marmotte_cuda_free(context, half))
    finally
        if other <> 0n then Assert.Equal(0, Native.marmotte_cuda_destroy(other))
        Assert.Equal(0, Native.marmotte_cuda_destroy(context))
