module Marmotte.Backends.Tests

open System
open System.Buffers.Binary
open System.Collections.Generic
open Marmotte.Core
open Marmotte.Cpu
open Marmotte.Cuda
open Marmotte.Metal
open Xunit

let private cpu () = new CpuBackend() :> ITensorBackend

let private download (backend: ITensorBackend) (tensor: ITensor) =
    let values = Array.zeroCreate<single> tensor.Shape.ElementCount
    backend.Download(tensor, values.AsMemory())
    values

/// Serves one matrix from memory, to build a WeightMatrix without a file format.
type private MatrixSource(encoding: TensorEncoding, rows: int, columns: int, payload: byte array) =
    let descriptor =
        { Name = "blk.0.attn_q.weight"
          Encoding = Some encoding
          SourceEncoding = string encoding
          Shape = [| uint64 rows; uint64 columns |]
          ElementCount = uint64 rows * uint64 columns
          ByteSize = Some(uint64 payload.Length) }

    interface ITensorSource with
        member _.Tensors = [| descriptor |] :> IReadOnlyList<TensorDescriptor>
        member _.TryFindTensor(name) = if name = descriptor.Name then Some descriptor else None
        member _.ReadPayload(_) = ReadOnlyMemory<byte>(payload)

    interface IDisposable with
        member _.Dispose() = ()

let private matrixOf encoding rows columns payload =
    use source = new MatrixSource(encoding, rows, columns, payload)
    WeightMatrix.Read(source, "blk.0.attn_q.weight")

/// A Q4_0 matrix of two rows: two 18-byte blocks per 64-value row.
let private weightMatrix () = matrixOf TensorEncoding.Q4_0 2 64 (Array.zeroCreate 72)

/// A one-layer, one-position cache of a single 32-value head.
let private cacheLayout () = KeyValueCacheLayout.Create(1, 1, 1, 32, TensorEncoding.F16, TensorEncoding.Q8_0)

[<Fact>]
let ``Add computes matrices scalars and empty tensors without modifying separate inputs`` () =
    use backend = cpu ()
    let cases =
        [ [| 2; 3 |], [| 1.0f; -2.0f; 3.0f; 4.0f; 0.5f; -6.0f |],
          [| 10.0f; 2.0f; -1.0f; 0.0f; 0.25f; 3.0f |], [| 11.0f; 0.0f; 2.0f; 4.0f; 0.75f; -3.0f |]
          [||], [| 1.5f |], [| -0.5f |], [| 1.0f |]
          [| 2; 0; 3 |], [||], [||], [||] ]

    for dimensions, leftValues, rightValues, expected in cases do
        let shape = TensorShape.Create dimensions
        use left = backend.Upload(shape, ReadOnlyMemory<single>(leftValues))
        use right = backend.Upload(TensorShape.Create dimensions, ReadOnlyMemory<single>(rightValues))
        use destination = backend.Allocate(shape)

        backend.Add(left, right, destination)
        backend.Synchronize()

        Assert.Equal<single>(expected, download backend destination)
        Assert.Equal<single>(leftValues, download backend left)
        Assert.Equal<single>(rightValues, download backend right)
        Assert.Equal(TensorEncoding.F32, destination.Encoding)
        Assert.Equal(backend.Backend, destination.Backend)

[<Theory>]
[<InlineData("left")>]
[<InlineData("right")>]
[<InlineData("both")>]
let ``Add supports exact in-place aliasing`` target =
    use backend = cpu ()
    let shape = TensorShape.Create [| 3 |]
    use left = backend.Upload(shape, ReadOnlyMemory<single>([| 1.0f; 2.0f; 3.0f |]))
    use right = backend.Upload(shape, ReadOnlyMemory<single>([| 4.0f; 5.0f; 6.0f |]))

    let destination, expected =
        match target with
        | "left" ->
            backend.Add(left, right, left)
            left, [| 5.0f; 7.0f; 9.0f |]
        | "right" ->
            backend.Add(left, right, right)
            right, [| 5.0f; 7.0f; 9.0f |]
        | _ ->
            backend.Add(left, left, left)
            left, [| 2.0f; 4.0f; 6.0f |]

    Assert.Equal<single>(expected, download backend destination)

[<Fact>]
let ``allocation clears storage and transfers do not expose aliases`` () =
    use backend = cpu ()
    let shape = TensorShape.Create [| 2 |]
    use zero = backend.Allocate(shape)
    Assert.Equal<single>([| 0.0f; 0.0f |], download backend zero)

    let hostValues = [| 1.0f; 2.0f |]
    use tensor = backend.Upload(shape, ReadOnlyMemory<single>(hostValues))
    hostValues[0] <- 99.0f
    let downloaded = download backend tensor
    downloaded[1] <- 99.0f
    Assert.Equal<single>([| 1.0f; 2.0f |], download backend tensor)

[<Fact>]
let ``transfers require exact lengths and validate before writing`` () =
    use backend = cpu ()
    let shape = TensorShape.Create [| 2 |]
    use tensor = backend.Allocate(shape)

    for length in [ 1; 3 ] do
        let buffer = Array.create length 42.0f
        Assert.Throws<ArgumentException>(fun () ->
            backend.Upload(shape, ReadOnlyMemory<single>(buffer)) |> ignore) |> ignore
        Assert.Throws<ArgumentException>(fun () -> backend.Download(tensor, buffer.AsMemory())) |> ignore
        Assert.All(buffer, fun value -> Assert.Equal(42.0f, value))

[<Fact>]
let ``Add rejects different shapes including equal element counts before writing`` () =
    use backend = cpu ()
    let shape = TensorShape.Create [| 2; 3 |]
    use left = backend.Allocate(shape)
    use right = backend.Allocate(shape)

    for dimensions in [ [| 3; 2 |]; [| 6 |]; [| 1; 3 |] ] do
        let otherShape = TensorShape.Create dimensions
        let sentinel = Array.create otherShape.ElementCount 42.0f
        use other = backend.Upload(otherShape, ReadOnlyMemory<single>(sentinel))
        Assert.Throws<ArgumentException>(fun () -> backend.Add(left, right, other)) |> ignore
        Assert.Equal<single>(sentinel, download backend other)

        use destination = backend.Upload(shape, ReadOnlyMemory<single>(Array.create 6 42.0f))
        Assert.Throws<ArgumentException>(fun () -> backend.Add(left, other, destination)) |> ignore
        Assert.Equal<single>(Array.create 6 42.0f, download backend destination)

[<Fact>]
let ``Add and Download reject tensors from another instance even of the same backend`` () =
    use backend = cpu ()
    use otherBackend = cpu ()
    let shape = TensorShape.Create [| 2 |]
    use local = backend.Upload(shape, ReadOnlyMemory<single>([| 7.0f; 8.0f |]))
    use foreign = otherBackend.Upload(shape, ReadOnlyMemory<single>([| 9.0f; 10.0f |]))

    for left, right, destination in [ foreign, local, local; local, foreign, local; local, local, foreign ] do
        Assert.Throws<ArgumentException>(fun () -> backend.Add(left, right, destination)) |> ignore
    Assert.Equal<single>([| 7.0f; 8.0f |], download backend local)
    Assert.Equal<single>([| 9.0f; 10.0f |], download otherBackend foreign)

    let output = [| 42.0f; 42.0f |]
    Assert.Throws<ArgumentException>(fun () -> backend.Download(foreign, output.AsMemory())) |> ignore
    Assert.Equal<single>([| 42.0f; 42.0f |], output)

[<Fact>]
let ``disposed tensors cannot be read or passed to Add`` () =
    use backend = cpu ()
    let shape = TensorShape.Create [| 2 |]
    use live = backend.Upload(shape, ReadOnlyMemory<single>([| 7.0f; 8.0f |]))
    let disposed = backend.Allocate(shape)
    disposed.Dispose()
    disposed.Dispose()

    for left, right, destination in [ disposed, live, live; live, disposed, live; live, live, disposed ] do
        Assert.Throws<ObjectDisposedException>(fun () -> backend.Add(left, right, destination)) |> ignore
    Assert.Throws<ObjectDisposedException>(fun () -> download backend disposed |> ignore) |> ignore
    Assert.Equal<single>([| 7.0f; 8.0f |], download backend live)
    Assert.Equal(2, disposed.Shape.ElementCount)

[<Fact>]
let ``UploadWeights keeps the stored encoding and matrix shape`` () =
    use backend = cpu ()
    use weights = backend.UploadWeights(weightMatrix ())

    Assert.Equal("blk.0.attn_q.weight", weights.Name)
    Assert.Equal(TensorEncoding.Q4_0, weights.Encoding)
    Assert.Equal(2, weights.Rows)
    Assert.Equal(64, weights.Columns)
    Assert.Equal(BackendKind.Cpu, weights.Backend)

[<Fact>]
let ``disposing the backend releases live tensors and makes their disposal a no-op`` () =
    let backend = cpu ()
    let shape = TensorShape.Create [| 2 |]
    let live = backend.Allocate(shape)
    let weights = backend.UploadWeights(weightMatrix ())
    let disposedFirst = backend.Upload(shape, ReadOnlyMemory<single>([| 1.0f; 2.0f |]))
    disposedFirst.Dispose()

    backend.Dispose()

    live.Dispose()
    live.Dispose()
    weights.Dispose()
    disposedFirst.Dispose()
    Assert.Equal(shape, live.Shape)
    Assert.Equal(64, weights.Columns)

[<Fact>]
let ``null shapes and tensors are rejected`` () =
    use backend = cpu ()
    let shape = TensorShape.Create [| 1 |]
    use tensor = backend.Allocate(shape)
    let nullShape = Unchecked.defaultof<TensorShape>
    let nullTensor = Unchecked.defaultof<ITensor>

    Assert.Throws<ArgumentNullException>(fun () -> backend.Allocate(nullShape) |> ignore) |> ignore
    Assert.Throws<ArgumentNullException>(fun () -> backend.Upload(nullShape, ReadOnlyMemory<single>.Empty) |> ignore) |> ignore
    Assert.Throws<ArgumentNullException>(fun () -> backend.Download(nullTensor, Memory<single>.Empty)) |> ignore
    Assert.Throws<ArgumentNullException>(fun () -> backend.UploadWeights(Unchecked.defaultof<WeightMatrix>) |> ignore) |> ignore
    Assert.Throws<ArgumentNullException>(fun () -> backend.MatVec(Unchecked.defaultof<IWeightTensor>, tensor, tensor)) |> ignore
    use weights = backend.UploadWeights(matrixOf TensorEncoding.F32 1 1 (Array.zeroCreate 4))
    Assert.Throws<ArgumentNullException>(fun () -> backend.MatVec(weights, nullTensor, tensor)) |> ignore
    Assert.Throws<ArgumentNullException>(fun () -> backend.MatVec(weights, tensor, nullTensor)) |> ignore
    Assert.Throws<ArgumentNullException>(fun () -> backend.MatMul(Unchecked.defaultof<IWeightTensor>, tensor, tensor)) |> ignore
    Assert.Throws<ArgumentNullException>(fun () -> backend.MatMul(weights, nullTensor, tensor)) |> ignore
    Assert.Throws<ArgumentNullException>(fun () -> backend.MatMul(weights, tensor, nullTensor)) |> ignore
    for left, right, destination in [ nullTensor, tensor, tensor; tensor, nullTensor, tensor; tensor, tensor, nullTensor ] do
        Assert.Throws<ArgumentNullException>(fun () -> backend.Add(left, right, destination)) |> ignore
        Assert.Throws<ArgumentNullException>(fun () -> backend.SwiGlu(left, right, destination)) |> ignore
        Assert.Throws<ArgumentNullException>(fun () -> backend.RmsNorm(left, right, 1e-5f, destination)) |> ignore

[<Fact>]
let ``Add matches a scalar reference for every vector tail length`` () =
    use backend = cpu ()
    let random = Random(1234)
    let randomValues length = Array.init length (fun _ -> float32 (random.NextDouble() * 200.0 - 100.0))

    // Lengths below, at and between multiples of 4, 8 and 16 lanes exercise vector blocks and tails.
    for length in 0 .. 67 do
        let shape = TensorShape.Create [| length |]
        let leftValues = randomValues length
        let rightValues = randomValues length
        let expected = Array.map2 (+) leftValues rightValues
        use left = backend.Upload(shape, ReadOnlyMemory<single>(leftValues))
        use right = backend.Upload(shape, ReadOnlyMemory<single>(rightValues))
        use destination = backend.Allocate(shape)

        backend.Add(left, right, destination)
        Assert.Equal<single>(expected, download backend destination)

        backend.Add(left, right, right)
        Assert.Equal<single>(expected, download backend right)

[<Fact>]
let ``Add preserves ordinary floating point special values in vector lanes and the tail`` () =
    use backend = cpu ()
    // 19 values place NaN and infinities in full vectors and in the scalar tail for 4, 8 or 16 lanes.
    let shape = TensorShape.Create [| 19 |]
    let leftValues = Array.init 19 (fun index -> if index % 3 = 0 then Single.NaN else Single.PositiveInfinity)
    let rightValues = Array.init 19 (fun index -> [| 1.0f; 2.0f; Single.NegativeInfinity |][index % 3])
    use left = backend.Upload(shape, ReadOnlyMemory<single>(leftValues))
    use right = backend.Upload(shape, ReadOnlyMemory<single>(rightValues))

    backend.Add(left, right, left)

    let result = download backend left
    for index in 0 .. 18 do
        match index % 3 with
        | 1 -> Assert.Equal(Single.PositiveInfinity, result[index])
        | _ -> Assert.True(Single.IsNaN(result[index]), $"Index {index} should be NaN.")

/// SwiGLU in double precision, independent of both Vector.Exp and MathF.Exp.
let private swiGluReference (gate: single) (up: single) =
    let x = float gate
    x / (1.0 + Math.Exp(-x)) * float up

/// About 8 units in the last place, plus an absolute floor for results that the F32 formula
/// flushes to zero when e^-x overflows.
let private assertClose (expected: float) (actual: single) =
    let tolerance = 1e-6 * abs expected + 1e-30
    Assert.True(abs (float actual - expected) <= tolerance, $"Expected {expected}, got {actual}.")

[<Fact>]
let ``SwiGlu matches a double-precision reference for every vector tail length`` () =
    use backend = cpu ()
    let random = Random(5678)
    let randomValues length range = Array.init length (fun _ -> float32 ((random.NextDouble() * 2.0 - 1.0) * range))

    // Lengths below, at and between multiples of 4, 8 and 16 lanes exercise vector blocks and tails.
    for length in 0 .. 67 do
        let shape = TensorShape.Create [| length |]
        let gateValues = randomValues length 30.0
        let upValues = randomValues length 5.0
        let expected = Array.map2 swiGluReference gateValues upValues
        use gate = backend.Upload(shape, ReadOnlyMemory<single>(gateValues))
        use up = backend.Upload(shape, ReadOnlyMemory<single>(upValues))
        use destination = backend.Allocate(shape)

        backend.SwiGlu(gate, up, destination)
        Array.iter2 assertClose expected (download backend destination)
        Assert.Equal<single>(gateValues, download backend gate)
        Assert.Equal<single>(upValues, download backend up)

        backend.SwiGlu(gate, up, up)
        Array.iter2 assertClose expected (download backend up)

        use upAgain = backend.Upload(shape, ReadOnlyMemory<single>(upValues))
        backend.SwiGlu(gate, upAgain, gate)
        Array.iter2 assertClose expected (download backend gate)

[<Fact>]
let ``SwiGlu handles extreme gates in vector lanes and the tail`` () =
    use backend = cpu ()
    // 21 values cycle through each case in full vectors and in the scalar tail for 4, 8 or 16 lanes.
    let cases = [| -100.0f; 100.0f; 0.0f; Single.NaN; Single.PositiveInfinity; Single.NegativeInfinity; -0.001f |]
    let gateValues = Array.init 21 (fun index -> cases[index % cases.Length])
    let shape = TensorShape.Create [| 21 |]
    use gate = backend.Upload(shape, ReadOnlyMemory<single>(gateValues))
    use up = backend.Upload(shape, ReadOnlyMemory<single>(Array.create 21 2.0f))

    backend.SwiGlu(gate, up, up)

    let result = download backend up
    for index in 0 .. 20 do
        let value = result[index]
        match gateValues[index] with
        | gateValue when Single.IsNaN gateValue -> Assert.True(Single.IsNaN value)
        | gateValue when Single.IsNegativeInfinity gateValue -> Assert.True(Single.IsNaN value)
        | gateValue when Single.IsPositiveInfinity gateValue -> Assert.Equal(Single.PositiveInfinity, value)
        | gateValue -> assertClose (swiGluReference gateValue 2.0f) value

[<Fact>]
let ``SwiGlu validates shapes ownership and disposal before writing`` () =
    use backend = cpu ()
    use otherBackend = cpu ()
    let shape = TensorShape.Create [| 2; 3 |]
    use gate = backend.Upload(shape, ReadOnlyMemory<single>(Array.create 6 1.0f))
    use up = backend.Upload(shape, ReadOnlyMemory<single>(Array.create 6 1.0f))
    use destination = backend.Upload(shape, ReadOnlyMemory<single>(Array.create 6 42.0f))
    use transposed = backend.Allocate(TensorShape.Create [| 3; 2 |])
    use foreign = otherBackend.Allocate(shape)
    let disposed = backend.Allocate(shape)
    disposed.Dispose()

    let shapeError = Assert.Throws<ArgumentException>(fun () -> backend.SwiGlu(gate, transposed, destination))
    Assert.Contains("[2; 3]", shapeError.Message)
    Assert.Contains("[3; 2]", shapeError.Message)
    Assert.Throws<ArgumentException>(fun () -> backend.SwiGlu(gate, up, transposed)) |> ignore
    Assert.Throws<ArgumentException>(fun () -> backend.SwiGlu(foreign, up, destination)) |> ignore
    Assert.Throws<ObjectDisposedException>(fun () -> backend.SwiGlu(gate, disposed, destination)) |> ignore
    Assert.Equal<single>(Array.create 6 42.0f, download backend destination)

/// RMS normalization of each row in double precision, independent of the kernel's summation order.
let private rmsNormReference (input: single array) (weight: single array) (epsilon: single) =
    let normalizeRow (row: single array) =
        let mean = row |> Array.averageBy (fun value -> float value * float value)
        let scale = 1.0 / Math.Sqrt(mean + float epsilon)
        Array.map2 (fun value weightValue -> float value * scale * float weightValue) row weight

    input |> Array.chunkBySize weight.Length |> Array.collect normalizeRow

[<Fact>]
let ``RmsNorm matches a double-precision reference for every row width`` () =
    use backend = cpu ()
    let random = Random(91011)
    let randomValues length low high = Array.init length (fun _ -> float32 (low + random.NextDouble() * (high - low)))

    // Widths below, at and between multiples of 4, 8 and 16 lanes exercise vector blocks and tails.
    for columns in 1 .. 67 do
        let shape = TensorShape.Create [| 3; columns |]
        let inputValues = randomValues (3 * columns) -10.0 10.0
        let weightValues = randomValues columns 0.5 1.5
        let expected = rmsNormReference inputValues weightValues 1e-5f
        use input = backend.Upload(shape, ReadOnlyMemory<single>(inputValues))
        use weight = backend.Upload(TensorShape.Create [| columns |], ReadOnlyMemory<single>(weightValues))
        use destination = backend.Allocate(shape)

        backend.RmsNorm(input, weight, 1e-5f, destination)
        Array.iter2 assertClose expected (download backend destination)
        Assert.Equal<single>(inputValues, download backend input)

        backend.RmsNorm(input, weight, 1e-5f, input)
        Array.iter2 assertClose expected (download backend input)

[<Fact>]
let ``RmsNorm normalizes Ministral-sized rows of any rank`` () =
    use backend = cpu ()
    let random = Random(1213)
    let inputValues = Array.init (4 * 3072) (fun _ -> float32 (random.NextDouble() * 2.0 - 1.0))
    let weightValues = Array.init 3072 (fun _ -> float32 (random.NextDouble() * 2.0))
    let expected = rmsNormReference inputValues weightValues 1e-5f
    use weight = backend.Upload(TensorShape.Create [| 3072 |], ReadOnlyMemory<single>(weightValues))

    for dimensions in [ [| 2; 2; 3072 |]; [| 4; 3072 |] ] do
        let shape = TensorShape.Create dimensions
        use input = backend.Upload(shape, ReadOnlyMemory<single>(inputValues))
        use destination = backend.Allocate(shape)
        backend.RmsNorm(input, weight, 1e-5f, destination)
        Array.iter2 assertClose expected (download backend destination)

    use vector = backend.Upload(TensorShape.Create [| 3072 |], ReadOnlyMemory<single>(inputValues[..3071]))
    backend.RmsNorm(vector, weight, 1e-5f, vector)
    Array.iter2 assertClose expected[..3071] (download backend vector)

[<Fact>]
let ``RmsNorm keeps huge zero and NaN rows independent`` () =
    use backend = cpu ()
    // Nine columns place values in full vectors and in the scalar tail. The squares of 1e30
    // exceed the F32 range, so the sum of squares must not be accumulated in F32.
    let huge = Array.init 9 (fun index -> if index % 2 = 0 then 1e30f else -1e30f)
    let zeros = Array.zeroCreate<single> 9
    let withNaN = Array.init 9 (fun index -> if index = 4 then Single.NaN else 1.0f)
    let weightValues = Array.init 9 (fun index -> float32 (index + 1))
    let shape = TensorShape.Create [| 3; 9 |]
    use input = backend.Upload(shape, ReadOnlyMemory<single>(Array.concat [ huge; zeros; withNaN ]))
    use weight = backend.Upload(TensorShape.Create [| 9 |], ReadOnlyMemory<single>(weightValues))

    backend.RmsNorm(input, weight, 1e-5f, input)

    let result = download backend input
    Array.iter2 assertClose (rmsNormReference huge weightValues 1e-5f) result[..8]
    Assert.Equal<single>(zeros, result[9..17])
    Assert.All(result[18..], fun value -> Assert.True(Single.IsNaN value))

[<Fact>]
let ``RmsNorm validates rank weight epsilon ownership and disposal before writing`` () =
    use backend = cpu ()
    use otherBackend = cpu ()
    let shape = TensorShape.Create [| 2; 3 |]
    use input = backend.Upload(shape, ReadOnlyMemory<single>(Array.create 6 1.0f))
    use weight = backend.Upload(TensorShape.Create [| 3 |], ReadOnlyMemory<single>(Array.create 3 1.0f))
    use destination = backend.Upload(shape, ReadOnlyMemory<single>(Array.create 6 42.0f))
    use scalar = backend.Allocate(TensorShape.Create [||])
    use scalarWeight = backend.Allocate(TensorShape.Create [| 1 |])
    use longWeight = backend.Allocate(TensorShape.Create [| 4 |])
    use matrixWeight = backend.Allocate(TensorShape.Create [| 1; 3 |])
    use transposed = backend.Allocate(TensorShape.Create [| 3; 2 |])
    use foreign = otherBackend.Allocate(TensorShape.Create [| 3 |])
    let disposed = backend.Allocate(shape)
    disposed.Dispose()

    Assert.Throws<ArgumentException>(fun () -> backend.RmsNorm(scalar, scalarWeight, 1e-5f, scalar)) |> ignore
    let weightError = Assert.Throws<ArgumentException>(fun () -> backend.RmsNorm(input, longWeight, 1e-5f, destination))
    Assert.Contains("[3]", weightError.Message)
    Assert.Contains("[4]", weightError.Message)
    Assert.Throws<ArgumentException>(fun () -> backend.RmsNorm(input, matrixWeight, 1e-5f, destination)) |> ignore
    Assert.Throws<ArgumentException>(fun () -> backend.RmsNorm(input, weight, 1e-5f, transposed)) |> ignore
    for epsilon in [ 0.0f; -1e-5f; Single.NaN; Single.PositiveInfinity ] do
        Assert.Throws<ArgumentOutOfRangeException>(fun () -> backend.RmsNorm(input, weight, epsilon, destination)) |> ignore
    Assert.Throws<ArgumentException>(fun () -> backend.RmsNorm(input, foreign, 1e-5f, destination)) |> ignore
    Assert.Throws<ObjectDisposedException>(fun () -> backend.RmsNorm(disposed, weight, 1e-5f, destination)) |> ignore
    Assert.Equal<single>(Array.create 6 42.0f, download backend destination)

[<Fact>]
let ``RmsNorm accepts empty tensors as no-ops`` () =
    use backend = cpu ()
    for dimensions, columns in [ [| 0; 3 |], 3; [| 2; 0 |], 0 ] do
        let shape = TensorShape.Create dimensions
        use input = backend.Allocate(shape)
        use weight = backend.Allocate(TensorShape.Create [| columns |])
        use destination = backend.Allocate(shape)
        backend.RmsNorm(input, weight, 1e-5f, destination)
        Assert.Empty(download backend destination)

/// Random weights in any encoding: floating-point values in [-1, 1], or random quant bytes whose
/// half-precision super-block scales are replaced by finite values between 1/256 and 1/16.
let private randomPayload (random: Random) (encoding: TensorEncoding) rows columns =
    let valueCount = rows * columns
    let randomFloats () = Array.init valueCount (fun _ -> float32 (random.NextDouble() * 2.0 - 1.0))
    let writeHalf (bytes: byte array) offset (value: single) =
        BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(offset, 2), BitConverter.HalfToUInt16Bits(Half.op_Explicit value))

    match encoding with
    | TensorEncoding.F32 ->
        let bytes = Array.zeroCreate<byte> (4 * valueCount)
        randomFloats () |> Array.iteri (fun index value -> BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(4 * index, 4), value))
        bytes
    | TensorEncoding.F16 ->
        let bytes = Array.zeroCreate<byte> (2 * valueCount)
        randomFloats () |> Array.iteri (fun index value -> writeHalf bytes (2 * index) value)
        bytes
    | TensorEncoding.BF16 ->
        let bytes = Array.zeroCreate<byte> (2 * valueCount)
        randomFloats ()
        |> Array.iteri (fun index value ->
            BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(2 * index, 2), uint16 (BitConverter.SingleToUInt32Bits value >>> 16)))
        bytes
    | _ ->
        let blockBytes = encoding.BlockBytes
        let bytes = Array.zeroCreate<byte> (valueCount / encoding.BlockElements * blockBytes)
        random.NextBytes(bytes)
        // Super-block scale offsets from the block layouts cited in docs/kernels.md: d first, plus
        // dmin for Q4_K and Q5_K; d and dmin last for Q2_K, d last for Q3_K and Q6_K.
        let scaleOffsets =
            match encoding with
            | TensorEncoding.Q4_1
            | TensorEncoding.Q5_1
            | TensorEncoding.Q4_K
            | TensorEncoding.Q5_K -> [ 0; 2 ]
            | TensorEncoding.Q2_K -> [ 80; 82 ]
            | TensorEncoding.Q3_K -> [ 108 ]
            | TensorEncoding.Q6_K -> [ 208 ]
            | _ -> [ 0 ]
        for block in 0 .. bytes.Length / blockBytes - 1 do
            for offset in scaleOffsets do
                writeHalf bytes (block * blockBytes + offset) (float32 (1.0 / 256.0 + random.NextDouble() / 16.0))
        bytes

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

[<Fact>]
let ``MatVec multiplies small F32 weights row by row`` () =
    use backend = cpu ()
    let weightValues = [| 1.0f; 2.0f; 3.0f; 4.0f; 5.0f; 6.0f |]
    let payload = weightValues |> Array.collect BitConverter.GetBytes
    use weights = backend.UploadWeights(matrixOf TensorEncoding.F32 2 3 payload)
    use input = backend.Upload(TensorShape.Create [| 3 |], ReadOnlyMemory<single>([| 1.0f; 0.5f; -1.0f |]))
    use destination = backend.Allocate(TensorShape.Create [| 2 |])

    backend.MatVec(weights, input, destination)

    Assert.Equal<single>([| -1.0f; 0.5f |], download backend destination)

[<Fact>]
let ``MatVec matches decoded rows in double precision for every encoding`` () =
    use backend = cpu ()
    let random = Random(1415)
    // Widths cover several blocks, and for floating-point weights the scalar tail of the dot product.
    let cases =
        [ TensorEncoding.F32, [ 1; 3; 67 ]
          TensorEncoding.F16, [ 5; 64 ]
          TensorEncoding.BF16, [ 7; 64 ]
          TensorEncoding.Q4_0, [ 32; 96 ]
          TensorEncoding.Q4_1, [ 32; 96 ]
          TensorEncoding.Q5_0, [ 32; 96 ]
          TensorEncoding.Q5_1, [ 32; 96 ]
          TensorEncoding.Q8_0, [ 32; 96 ]
          TensorEncoding.IQ4_NL, [ 32; 96 ]
          TensorEncoding.Q2_K, [ 256; 512 ]
          TensorEncoding.Q3_K, [ 256; 512 ]
          TensorEncoding.Q4_K, [ 256; 512 ]
          TensorEncoding.Q5_K, [ 256; 512 ]
          TensorEncoding.Q6_K, [ 256; 512 ] ]

    for encoding, widths in cases do
        for columns in widths do
            let matrix = matrixOf encoding 7 columns (randomPayload random encoding 7 columns)
            let inputValues = Array.init columns (fun _ -> float32 (random.NextDouble() * 2.0 - 1.0))
            assertMatVec backend matrix inputValues

[<Fact>]
let ``MatVec handles Ministral-sized Q4_K and Q6_K rows`` () =
    use backend = cpu ()
    let random = Random(1617)
    let inputValues = Array.init 3072 (fun _ -> float32 (random.NextDouble() * 2.0 - 1.0))
    for encoding in [ TensorEncoding.Q4_K; TensorEncoding.Q6_K ] do
        assertMatVec backend (matrixOf encoding 16 3072 (randomPayload random encoding 16 3072)) inputValues

[<Fact>]
let ``MatVec validates shapes aliasing ownership and disposal before writing`` () =
    use backend = cpu ()
    use otherBackend = cpu ()
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
    use foreignInput = otherBackend.Allocate(TensorShape.Create [| 4 |])
    let disposedInput = vector ()
    disposedInput.Dispose()

    let shapeError = Assert.Throws<ArgumentException>(fun () -> backend.MatVec(weights, shortInput, destination))
    Assert.Contains("[4]", shapeError.Message)
    Assert.Contains("[3]", shapeError.Message)
    Assert.Throws<ArgumentException>(fun () -> backend.MatVec(weights, input, longDestination)) |> ignore
    Assert.Throws<ArgumentException>(fun () -> backend.MatVec(weights, matrixInput, destination)) |> ignore
    Assert.Throws<ArgumentException>(fun () -> backend.MatVec(weights, input, input)) |> ignore
    Assert.Throws<ArgumentException>(fun () -> backend.MatVec(foreignWeights, input, destination)) |> ignore
    Assert.Throws<ArgumentException>(fun () -> backend.MatVec(weights, foreignInput, destination)) |> ignore
    Assert.Throws<ObjectDisposedException>(fun () -> backend.MatVec(disposedWeights, input, destination)) |> ignore
    Assert.Throws<ObjectDisposedException>(fun () -> backend.MatVec(weights, disposedInput, destination)) |> ignore
    Assert.Equal<single>(Array.create 4 42.0f, download backend destination)

/// Checks MatMul token by token against decoded rows multiplied in double precision, with the
/// same tolerance as MatVec: relative to the sum of absolute products of each output.
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
let ``MatMul matches decoded rows in double precision for every encoding and batch size`` () =
    use backend = cpu ()
    let random = Random(1819)
    let cases =
        [ TensorEncoding.F32, 67
          TensorEncoding.F16, 64
          TensorEncoding.BF16, 64
          TensorEncoding.Q4_0, 96
          TensorEncoding.Q4_1, 96
          TensorEncoding.Q5_0, 96
          TensorEncoding.Q5_1, 96
          TensorEncoding.Q8_0, 96
          TensorEncoding.IQ4_NL, 96
          TensorEncoding.Q2_K, 512
          TensorEncoding.Q3_K, 512
          TensorEncoding.Q4_K, 512
          TensorEncoding.Q5_K, 512
          TensorEncoding.Q6_K, 512 ]

    for encoding, columns in cases do
        let matrix = matrixOf encoding 7 columns (randomPayload random encoding 7 columns)
        for tokens in [ 1; 2; 5 ] do
            let inputValues = Array.init (tokens * columns) (fun _ -> float32 (random.NextDouble() * 2.0 - 1.0))
            assertMatMul backend matrix tokens inputValues

[<Fact>]
let ``MatMul gives each token exactly its MatVec result`` () =
    use backend = cpu ()
    let random = Random(2021)
    let matrix = matrixOf TensorEncoding.Q4_K 16 3072 (randomPayload random TensorEncoding.Q4_K 16 3072)
    let inputValues = Array.init (4 * 3072) (fun _ -> float32 (random.NextDouble() * 2.0 - 1.0))
    use weights = backend.UploadWeights(matrix)
    use input = backend.Upload(TensorShape.Create [| 4; 3072 |], ReadOnlyMemory<single>(inputValues))
    use destination = backend.Allocate(TensorShape.Create [| 4; 16 |])

    backend.MatMul(weights, input, destination)

    let batched = download backend destination
    for token in 0 .. 3 do
        use vector = backend.Upload(TensorShape.Create [| 3072 |], ReadOnlyMemory<single>(inputValues, token * 3072, 3072))
        use output = backend.Allocate(TensorShape.Create [| 16 |])
        backend.MatVec(weights, vector, output)
        Assert.Equal<single>(download backend output, batched[token * 16 .. token * 16 + 15])

[<Fact>]
let ``MatMul accepts an empty batch as a no-op`` () =
    use backend = cpu ()
    use weights = backend.UploadWeights(matrixOf TensorEncoding.F32 2 3 (Array.zeroCreate 24))
    use input = backend.Allocate(TensorShape.Create [| 0; 3 |])
    use destination = backend.Allocate(TensorShape.Create [| 0; 2 |])

    backend.MatMul(weights, input, destination)

    Assert.Empty(download backend destination)

[<Fact>]
let ``MatMul validates shapes aliasing ownership and disposal before writing`` () =
    use backend = cpu ()
    use otherBackend = cpu ()
    let matrix = matrixOf TensorEncoding.F32 4 4 (Array.zeroCreate 64)
    use weights = backend.UploadWeights(matrix)
    use foreignWeights = otherBackend.UploadWeights(matrix)
    let disposedWeights = backend.UploadWeights(matrix)
    disposedWeights.Dispose()
    let batch () = backend.Upload(TensorShape.Create [| 2; 4 |], ReadOnlyMemory<single>(Array.create 8 1.0f))
    use input = batch ()
    use destination = backend.Upload(TensorShape.Create [| 2; 4 |], ReadOnlyMemory<single>(Array.create 8 42.0f))
    use vector = backend.Allocate(TensorShape.Create [| 4 |])
    use narrowInput = backend.Allocate(TensorShape.Create [| 2; 3 |])
    use otherTokenCount = backend.Allocate(TensorShape.Create [| 3; 4 |])
    use wideDestination = backend.Allocate(TensorShape.Create [| 2; 5 |])
    use foreignInput = otherBackend.Allocate(TensorShape.Create [| 2; 4 |])
    let disposedInput = batch ()
    disposedInput.Dispose()

    let shapeError = Assert.Throws<ArgumentException>(fun () -> backend.MatMul(weights, narrowInput, destination))
    Assert.Contains("[tokens; 4]", shapeError.Message)
    Assert.Contains("[2; 3]", shapeError.Message)
    Assert.Throws<ArgumentException>(fun () -> backend.MatMul(weights, vector, destination)) |> ignore
    Assert.Throws<ArgumentException>(fun () -> backend.MatMul(weights, input, otherTokenCount)) |> ignore
    Assert.Throws<ArgumentException>(fun () -> backend.MatMul(weights, input, wideDestination)) |> ignore
    Assert.Throws<ArgumentException>(fun () -> backend.MatMul(weights, input, input)) |> ignore
    Assert.Throws<ArgumentException>(fun () -> backend.MatMul(foreignWeights, input, destination)) |> ignore
    Assert.Throws<ArgumentException>(fun () -> backend.MatMul(weights, foreignInput, destination)) |> ignore
    Assert.Throws<ObjectDisposedException>(fun () -> backend.MatMul(disposedWeights, input, destination)) |> ignore
    Assert.Throws<ObjectDisposedException>(fun () -> backend.MatMul(weights, disposedInput, destination)) |> ignore
    Assert.Equal<single>(Array.create 8 42.0f, download backend destination)

let private createBackend name =
    match name with
    | "CPU" -> cpu ()
    | "CUDA" -> new CudaBackend() :> ITensorBackend
    | "Metal" -> new MetalBackend() :> ITensorBackend
    | _ -> invalidArg (nameof name) "Unknown test backend."

[<Theory>]
[<InlineData("CUDA")>]
[<InlineData("Metal")>]
let ``GPU operations explicitly fail without requiring GPU hardware`` name =
    use backend = createBackend name
    use host = cpu ()
    let shape = TensorShape.Create [| 1 |]
    use tensor = host.Allocate(shape)
    use weights = host.UploadWeights(weightMatrix ())
    use cache = host.AllocateKeyValueCache(cacheLayout ())
    let operations =
        [ (fun () -> backend.Allocate(shape) |> ignore)
          (fun () -> backend.Upload(shape, ReadOnlyMemory<single>([| 1.0f |])) |> ignore)
          (fun () -> backend.UploadWeights(weightMatrix ()) |> ignore)
          (fun () -> backend.AllocateKeyValueCache(cacheLayout ()) |> ignore)
          (fun () -> backend.Download(tensor, Memory<single>([| 0.0f |])))
          (fun () -> backend.Synchronize())
          (fun () -> backend.Add(tensor, tensor, tensor))
          (fun () -> backend.SwiGlu(tensor, tensor, tensor))
          (fun () -> backend.RmsNorm(tensor, tensor, 1e-5f, tensor))
          (fun () -> backend.EmbeddingLookup(weights, ReadOnlyMemory<int>([| 0 |]), tensor))
          (fun () -> backend.MatVec(weights, tensor, tensor))
          (fun () -> backend.MatMul(weights, tensor, tensor))
          (fun () -> backend.Rope(tensor, ReadOnlyMemory<int>([| 0 |]), RotaryEmbedding.Standard(2, 10000.0, RopePairing.AdjacentPairs), tensor))
          (fun () -> backend.ApplyAttentionTemperature(tensor, ReadOnlyMemory<int>([| 0 |]), AttentionTemperature.Create(0.1, 16384), tensor))
          (fun () -> backend.StoreKeyValues(cache, 0, 0, tensor, tensor))
          (fun () -> backend.ReadKeyValues(cache, 0, 0, tensor, tensor)) ]

    let expectedKind = if name = "CUDA" then BackendKind.Cuda else BackendKind.Metal
    Assert.Equal(expectedKind, backend.Backend)
    for operation in operations do
        let error = Assert.Throws<NotImplementedException>(fun () -> operation ())
        Assert.Contains(name, error.Message)

[<Theory>]
[<InlineData("CPU")>]
[<InlineData("CUDA")>]
[<InlineData("Metal")>]
let ``disposed backends reject all operations and allow repeated disposal`` name =
    let backend = createBackend name
    use host = cpu ()
    let shape = TensorShape.Create [| 1 |]
    use tensor = host.Allocate(shape)
    use weights = host.UploadWeights(weightMatrix ())
    use cache = host.AllocateKeyValueCache(cacheLayout ())
    backend.Dispose()
    backend.Dispose()

    Assert.Throws<ObjectDisposedException>(fun () -> backend.Allocate(shape) |> ignore) |> ignore
    Assert.Throws<ObjectDisposedException>(fun () -> backend.Upload(shape, ReadOnlyMemory<single>.Empty) |> ignore) |> ignore
    Assert.Throws<ObjectDisposedException>(fun () -> backend.UploadWeights(weightMatrix ()) |> ignore) |> ignore
    Assert.Throws<ObjectDisposedException>(fun () -> backend.Download(tensor, Memory<single>.Empty)) |> ignore
    Assert.Throws<ObjectDisposedException>(fun () -> backend.Synchronize()) |> ignore
    Assert.Throws<ObjectDisposedException>(fun () -> backend.Add(tensor, tensor, tensor)) |> ignore
    Assert.Throws<ObjectDisposedException>(fun () -> backend.SwiGlu(tensor, tensor, tensor)) |> ignore
    Assert.Throws<ObjectDisposedException>(fun () -> backend.RmsNorm(tensor, tensor, 1e-5f, tensor)) |> ignore
    Assert.Throws<ObjectDisposedException>(fun () -> backend.EmbeddingLookup(weights, ReadOnlyMemory<int>([| 0 |]), tensor)) |> ignore
    Assert.Throws<ObjectDisposedException>(fun () -> backend.MatVec(weights, tensor, tensor)) |> ignore
    Assert.Throws<ObjectDisposedException>(fun () -> backend.MatMul(weights, tensor, tensor)) |> ignore
    Assert.Throws<ObjectDisposedException>(fun () -> backend.Rope(tensor, ReadOnlyMemory<int>([| 0 |]), RotaryEmbedding.Standard(2, 10000.0, RopePairing.AdjacentPairs), tensor)) |> ignore
    Assert.Throws<ObjectDisposedException>(fun () -> backend.ApplyAttentionTemperature(tensor, ReadOnlyMemory<int>([| 0 |]), AttentionTemperature.Create(0.1, 16384), tensor)) |> ignore
    Assert.Throws<ObjectDisposedException>(fun () -> backend.AllocateKeyValueCache(cacheLayout ()) |> ignore) |> ignore
    Assert.Throws<ObjectDisposedException>(fun () -> backend.StoreKeyValues(cache, 0, 0, tensor, tensor)) |> ignore
    Assert.Throws<ObjectDisposedException>(fun () -> backend.ReadKeyValues(cache, 0, 0, tensor, tensor)) |> ignore
