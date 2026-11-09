module Marmotte.Backends.MetalTests

open System
open Marmotte.Core
open Marmotte.Cpu
open Marmotte.Metal
open Xunit

/// Explicit opt-in: when enabled, a missing library or device fails the tests.
type MetalFactAttribute() as this =
    inherit FactAttribute()
    do
        if Environment.GetEnvironmentVariable("MARMOTTE_TEST_METAL") <> "1" then
            this.Skip <- "Set MARMOTTE_TEST_METAL=1 and deploy the native library; see docs/metal.md."

let private download (backend: ITensorBackend) (tensor: ITensor) =
    let values = Array.zeroCreate<single> tensor.Shape.ElementCount
    backend.Download(tensor, values.AsMemory())
    values

let private upload (backend: ITensorBackend) (shape: TensorShape) (values: single array) =
    backend.Upload(shape, ReadOnlyMemory<single>(values))

[<Fact>]
let ``Metal empty tensors need no runtime and still enforce shape ownership and lifetime`` () =
    use backend = new MetalBackend() :> ITensorBackend
    use otherBackend = new MetalBackend() :> ITensorBackend
    use cpu = new CpuBackend() :> ITensorBackend
    let shape = TensorShape.Create [| 2; 0; 3 |]
    use left = backend.Allocate(shape)
    use right = backend.Upload(shape, ReadOnlyMemory<single>.Empty)
    use destination = backend.Allocate(shape)
    use differentShape = backend.Allocate(TensorShape.Create [| 0 |])
    use foreign = otherBackend.Allocate(shape)
    use host = cpu.Allocate(shape)

    backend.Add(left, right, destination)
    backend.Add(left, left, left)
    backend.Synchronize()
    Assert.Empty(download backend destination)
    Assert.Equal(BackendKind.Metal, destination.Backend)
    Assert.Equal(TensorEncoding.F32, destination.Encoding)

    for invalid in [ differentShape; foreign; host ] do
        for a, b, output in [ invalid, right, destination; left, invalid, destination; left, right, invalid ] do
            Assert.Throws<ArgumentException>(fun () -> backend.Add(a, b, output)) |> ignore

    for tensor in [ foreign; host ] do
        Assert.Throws<ArgumentException>(fun () -> backend.Download(tensor, Memory<single>.Empty)) |> ignore

    left.Dispose()
    left.Dispose()
    Assert.Throws<ObjectDisposedException>(fun () -> backend.Add(left, right, destination)) |> ignore
    Assert.Throws<ObjectDisposedException>(fun () -> backend.Add(right, left, destination)) |> ignore
    Assert.Throws<ObjectDisposedException>(fun () -> backend.Add(right, right, left)) |> ignore
    Assert.Throws<ObjectDisposedException>(fun () -> backend.Download(left, Memory<single>.Empty)) |> ignore
    backend.Dispose()
    destination.Dispose()
    right.Dispose()
    Assert.Equal(shape, destination.Shape)
    Assert.Throws<ObjectDisposedException>(fun () -> backend.Synchronize()) |> ignore

[<Fact>]
let ``Metal validates arguments before native loading or host writes`` () =
    use backend = new MetalBackend() :> ITensorBackend
    let nullShape = Unchecked.defaultof<TensorShape>
    let nullTensor = Unchecked.defaultof<ITensor>
    let emptyShape = TensorShape.Create [| 0 |]
    use empty = backend.Allocate(emptyShape)
    Assert.Throws<ArgumentNullException>(fun () -> backend.Allocate(nullShape) |> ignore) |> ignore
    Assert.Throws<ArgumentNullException>(fun () -> backend.Upload(nullShape, ReadOnlyMemory<single>.Empty) |> ignore) |> ignore
    Assert.Throws<ArgumentException>(fun () -> backend.Upload(TensorShape.Create [| 1 |], ReadOnlyMemory<single>.Empty) |> ignore) |> ignore

    for a, b, output in [ nullTensor, empty, empty; empty, nullTensor, empty; empty, empty, nullTensor ] do
        Assert.Throws<ArgumentNullException>(fun () -> backend.Add(a, b, output)) |> ignore

    let sentinel = [| 42.0f |]
    Assert.Throws<ArgumentNullException>(fun () -> backend.Download(nullTensor, sentinel.AsMemory())) |> ignore
    Assert.Throws<ArgumentException>(fun () -> backend.Download(empty, sentinel.AsMemory())) |> ignore
    Assert.Equal<single>([| 42.0f |], sentinel)

[<MetalFact>]
[<Trait("Category", "Metal")>]
let ``Metal Add matches CPU for scalars matrices and threadgroup boundaries`` () =
    use backend = new MetalBackend() :> ITensorBackend
    use cpu = new CpuBackend() :> ITensorBackend
    let shapes = [ [||]; [| 0 |]; [| 2; 3 |]; [| 31 |]; [| 32 |]; [| 33 |]; [| 63 |]; [| 64 |]; [| 65 |]; [| 65537 |] ]

    for dimensions in shapes do
        let shape = TensorShape.Create dimensions
        let leftValues = Array.init shape.ElementCount (fun i -> single (i % 127 - 63) / 8.0f)
        let rightValues = Array.init shape.ElementCount (fun i -> single (i % 17) / 4.0f)
        use left = upload backend shape leftValues
        use right = upload backend shape rightValues
        use output = backend.Allocate(shape)
        Assert.All(download backend output, fun value -> Assert.Equal(0.0f, value))
        use cpuLeft = upload cpu shape leftValues
        use cpuRight = upload cpu shape rightValues
        use cpuOutput = cpu.Allocate(shape)

        backend.Add(left, right, output)
        cpu.Add(cpuLeft, cpuRight, cpuOutput)
        Assert.Equal<single>(download cpu cpuOutput, download backend output)
        Assert.Equal<single>(leftValues, download backend left)
        Assert.Equal<single>(rightValues, download backend right)

[<MetalFact>]
[<Trait("Category", "Metal")>]
let ``Metal Add follows IEEE 754 for NaN infinities and signed zero`` () =
    use backend = new MetalBackend() :> ITensorBackend
    let leftSpecial = [| Single.NaN; Single.PositiveInfinity; Single.PositiveInfinity; -0.0f; 1.0f |]
    let rightSpecial = [| 1.0f; 2.0f; Single.NegativeInfinity; -0.0f; -1.0f |]
    let shape = TensorShape.Create [| leftSpecial.Length |]
    use left = upload backend shape leftSpecial
    use right = upload backend shape rightSpecial
    backend.Add(left, right, left)
    let actual = download backend left
    Assert.True(Single.IsNaN(actual[0]))
    Assert.Equal(Single.PositiveInfinity, actual[1])
    Assert.True(Single.IsNaN(actual[2]))
    Assert.Equal(BitConverter.SingleToInt32Bits(-0.0f), BitConverter.SingleToInt32Bits(actual[3]))
    Assert.Equal(BitConverter.SingleToInt32Bits(0.0f), BitConverter.SingleToInt32Bits(actual[4]))

// The GPU may flush subnormal values to zero, unlike the CPU; see docs/metal.md. Transfers never do.
[<MetalFact>]
[<Trait("Category", "Metal")>]
let ``Metal Add computes subnormals exactly or flushes them to zero and transfers preserve their bits`` () =
    use backend = new MetalBackend() :> ITensorBackend
    let largestSubnormal = BitConverter.Int32BitsToSingle 0x007FFFFF
    let smallestNormal = BitConverter.Int32BitsToSingle 0x00800000
    let inputs = [| Single.Epsilon; largestSubnormal |]
    let exactSums = [| 2.0f * Single.Epsilon; smallestNormal |]
    let shape = TensorShape.Create [| inputs.Length |]
    use left = upload backend shape inputs
    use right = upload backend shape [| Single.Epsilon; Single.Epsilon |]
    use output = backend.Allocate(shape)

    Assert.Equal<int>(Array.map BitConverter.SingleToInt32Bits inputs, Array.map BitConverter.SingleToInt32Bits (download backend left))

    backend.Add(left, right, output)
    let sums = download backend output

    for actual, exact in Array.zip sums exactSums do
        Assert.True((actual = exact || actual = 0.0f), $"Expected {exact} or zero, not {actual}.")

    Assert.Equal<int>(Array.map BitConverter.SingleToInt32Bits inputs, Array.map BitConverter.SingleToInt32Bits (download backend left))

[<MetalFact>]
[<Trait("Category", "Metal")>]
let ``Metal queues dependent in-place operations and owns transfer buffers`` () =
    use backend = new MetalBackend() :> ITensorBackend
    let shape = TensorShape.Create [| 257 |]
    let host = Array.create 259 2.0f
    use left = backend.Upload(shape, ReadOnlyMemory<single>(host, 1, 257))
    Array.Fill(host, 99.0f)
    use right = upload backend shape (Array.create 257 3.0f)
    backend.Add(left, right, left)
    backend.Add(left, right, right)
    backend.Add(right, right, right)

    let result = Array.create 259 -1.0f
    backend.Download(right, Memory<single>(result, 1, 257))
    Assert.Equal(-1.0f, result[0])
    Assert.Equal(-1.0f, result[258])
    Assert.All(result[1..257], fun value -> Assert.Equal(16.0f, value))
    result[1] <- 999.0f
    Assert.All(download backend right, fun value -> Assert.Equal(16.0f, value))
    Assert.All(download backend left, fun value -> Assert.Equal(5.0f, value))
    backend.Synchronize()

// More submissions than a command queue accepts at once, by default 64: the submitting
// thread must wait for capacity, without losing the order of dependent additions.
[<MetalFact>]
[<Trait("Category", "Metal")>]
let ``Metal keeps submission order beyond the capacity of its command queue`` () =
    use backend = new MetalBackend() :> ITensorBackend
    let shape = TensorShape.Create [| 1025 |]
    use one = upload backend shape (Array.create shape.ElementCount 1.0f)
    use total = backend.Allocate(shape)

    for _ in 1 .. 500 do
        backend.Add(total, one, total)

    Assert.All(download backend total, fun value -> Assert.Equal(500.0f, value))

[<MetalFact>]
[<Trait("Category", "Metal")>]
let ``Metal rejects invalid device tensors before changing destination`` () =
    use backend = new MetalBackend() :> ITensorBackend
    use otherBackend = new MetalBackend() :> ITensorBackend
    use cpu = new CpuBackend() :> ITensorBackend
    let shape = TensorShape.Create [| 2; 3 |]
    use left = upload backend shape (Array.create 6 2.0f)
    use destination = upload backend shape (Array.create 6 42.0f)
    use foreign = otherBackend.Allocate(shape)
    use host = cpu.Allocate(shape)
    use disposed = backend.Allocate(shape)
    disposed.Dispose()

    for dimensions in [ [| 6 |]; [| 3; 2 |]; [| 1; 3 |] ] do
        use differentShape = backend.Allocate(TensorShape.Create dimensions)
        Assert.Throws<ArgumentException>(fun () -> backend.Add(left, differentShape, destination)) |> ignore
        Assert.Throws<ArgumentException>(fun () -> backend.Add(left, left, differentShape)) |> ignore

    for invalid in [ foreign; host ] do
        for a, b, output in [ invalid, left, destination; left, invalid, destination; left, left, invalid ] do
            Assert.Throws<ArgumentException>(fun () -> backend.Add(a, b, output)) |> ignore

    Assert.Throws<ObjectDisposedException>(fun () -> backend.Add(left, disposed, destination)) |> ignore
    Assert.All(download backend destination, fun value -> Assert.Equal(42.0f, value))
    let sentinel = Array.create 6 77.0f
    Assert.Throws<ArgumentException>(fun () -> backend.Download(foreign, sentinel.AsMemory())) |> ignore
    Assert.Throws<ObjectDisposedException>(fun () -> backend.Download(disposed, sentinel.AsMemory())) |> ignore
    Assert.All(sentinel, fun value -> Assert.Equal(77.0f, value))

    for length in [ 5; 7 ] do
        let values = Array.create length 77.0f
        Assert.Throws<ArgumentException>(fun () -> backend.Upload(shape, ReadOnlyMemory<single>(values)) |> ignore) |> ignore
        Assert.Throws<ArgumentException>(fun () -> backend.Download(left, values.AsMemory())) |> ignore
        Assert.All(values, fun value -> Assert.Equal(77.0f, value))

[<MetalFact>]
[<Trait("Category", "Metal")>]
let ``Metal tensor disposal keeps queued work valid and backend disposal releases only its own resources`` () =
    use backend = new MetalBackend() :> ITensorBackend
    use otherBackend = new MetalBackend() :> ITensorBackend
    let shape = TensorShape.Create [| 65537 |]
    use input = upload backend shape (Array.create shape.ElementCount 3.0f)
    use output = backend.Allocate(shape)
    use survivor = upload otherBackend shape (Array.create shape.ElementCount 7.0f)

    backend.Add(input, input, output)
    input.Dispose()
    Assert.All(download backend output, fun value -> Assert.Equal(6.0f, value))

    backend.Add(output, output, output)
    backend.Dispose()
    backend.Dispose()
    output.Dispose()
    input.Dispose()
    Assert.Equal(shape, output.Shape)
    Assert.Throws<ObjectDisposedException>(fun () -> backend.Download(output, Memory<single>(Array.zeroCreate shape.ElementCount))) |> ignore

    otherBackend.Add(survivor, survivor, survivor)
    Assert.All(download otherBackend survivor, fun value -> Assert.Equal(14.0f, value))

// SwiGLU in double precision, independent of both Metal's exp and the CPU's Vector.Exp.
let private swiGluReference (gate: single) (up: single) =
    let x = float gate
    x / (1.0 + Math.Exp(-x)) * float up

// A relative tolerance of about 8 units in the last place, plus an absolute floor for results
// the GPU may flush to zero: F32 subnormals and e^-x overflow for gates below about -88.7.
let private assertSwiGluClose (expected: float) (actual: single) =
    let tolerance = 1e-6 * abs expected + 1e-30
    Assert.True(abs (float actual - expected) <= tolerance, $"Expected {expected}, got {actual}.")

[<Fact>]
let ``Metal SwiGlu validates empty tensors and mixed encodings without loading the runtime`` () =
    use backend = new MetalBackend() :> ITensorBackend
    use otherBackend = new MetalBackend() :> ITensorBackend
    let shape = TensorShape.Create [| 2; 0; 3 |]
    use gate = backend.Allocate(shape)
    use up = backend.Allocate(shape)
    use destination = backend.Allocate(shape)
    use halfGate = backend.Allocate(shape, TensorEncoding.F16)
    use halfUp = backend.Allocate(shape, TensorEncoding.F16)
    use halfDestination = backend.Allocate(shape, TensorEncoding.F16)
    use foreign = otherBackend.Allocate(shape)

    backend.SwiGlu(gate, up, destination)
    backend.SwiGlu(gate, up, up)
    backend.SwiGlu(gate, gate, gate)
    backend.SwiGlu(halfGate, halfUp, halfDestination)
    backend.SwiGlu(halfGate, halfUp, halfUp)
    backend.Synchronize()
    Assert.Empty(download backend destination)

    // Mixed encodings are rejected before any submission, even for empty tensors.
    for g, u, d in [ halfGate, up, destination; gate, halfUp, destination; gate, up, halfDestination ] do
        Assert.Throws<ArgumentException>(fun () -> backend.SwiGlu(g, u, d)) |> ignore
    for g, u, d in [ foreign, up, destination; gate, foreign, destination; gate, up, foreign ] do
        Assert.Throws<ArgumentException>(fun () -> backend.SwiGlu(g, u, d)) |> ignore

[<MetalFact>]
[<Trait("Category", "Metal")>]
let ``Metal SwiGlu matches a double-precision reference for every vector tail length`` () =
    use backend = new MetalBackend() :> ITensorBackend
    let random = Random(5678)
    let randomValues length range = Array.init length (fun _ -> float32 ((random.NextDouble() * 2.0 - 1.0) * range))

    // Lengths below, at and between multiples of 4, 8 and 16 lanes exercise vector blocks and tails.
    for length in 0 .. 67 do
        let shape = TensorShape.Create [| length |]
        let gateValues = randomValues length 30.0
        let upValues = randomValues length 5.0
        let expected = Array.map2 swiGluReference gateValues upValues
        use gate = upload backend shape gateValues
        use up = upload backend shape upValues
        use destination = backend.Allocate(shape)

        backend.SwiGlu(gate, up, destination)
        Array.iter2 assertSwiGluClose expected (download backend destination)
        Assert.Equal<single>(gateValues, download backend gate)
        Assert.Equal<single>(upValues, download backend up)

        // Reuse either input as the destination; the other input is still the original operand.
        backend.SwiGlu(gate, up, up)
        Array.iter2 assertSwiGluClose expected (download backend up)

        use upAgain = upload backend shape upValues
        backend.SwiGlu(gate, upAgain, gate)
        Array.iter2 assertSwiGluClose expected (download backend gate)

[<MetalFact>]
[<Trait("Category", "Metal")>]
let ``Metal SwiGlu handles extreme gates in vector lanes and the tail`` () =
    use backend = new MetalBackend() :> ITensorBackend
    // 21 values cycle through each case in full threadgroups and in the partial tail.
    let cases = [| -100.0f; 100.0f; 0.0f; Single.NaN; Single.PositiveInfinity; Single.NegativeInfinity; -0.001f |]
    let gateValues = Array.init 21 (fun index -> cases[index % cases.Length])
    let shape = TensorShape.Create [| 21 |]
    use gate = upload backend shape gateValues
    use up = upload backend shape (Array.create 21 2.0f)

    backend.SwiGlu(gate, up, up)

    let result = download backend up
    for index in 0 .. 20 do
        let value = result[index]
        match gateValues[index] with
        | gateValue when Single.IsNaN gateValue -> Assert.True(Single.IsNaN value)
        | gateValue when Single.IsNegativeInfinity gateValue -> Assert.True(Single.IsNaN value)
        | gateValue when Single.IsPositiveInfinity gateValue -> Assert.Equal(Single.PositiveInfinity, value)
        | gateValue -> assertSwiGluClose (swiGluReference gateValue 2.0f) value

[<MetalFact>]
[<Trait("Category", "Metal")>]
let ``Metal SwiGlu validates shapes encodings ownership and disposal before writing`` () =
    use backend = new MetalBackend() :> ITensorBackend
    use otherBackend = new MetalBackend() :> ITensorBackend
    let shape = TensorShape.Create [| 2; 3 |]
    use gate = upload backend shape (Array.create 6 1.0f)
    use up = upload backend shape (Array.create 6 1.0f)
    use destination = upload backend shape (Array.create 6 42.0f)
    use transposed = backend.Allocate(TensorShape.Create [| 3; 2 |])
    use foreign = otherBackend.Allocate(shape)
    use half = backend.Allocate(shape, TensorEncoding.F16)
    let disposed = backend.Allocate(shape)
    disposed.Dispose()

    let shapeError = Assert.Throws<ArgumentException>(fun () -> backend.SwiGlu(gate, transposed, destination))
    Assert.Contains("[2; 3]", shapeError.Message)
    Assert.Contains("[3; 2]", shapeError.Message)
    Assert.Throws<ArgumentException>(fun () -> backend.SwiGlu(gate, up, transposed)) |> ignore
    Assert.Throws<ArgumentException>(fun () -> backend.SwiGlu(foreign, up, destination)) |> ignore
    Assert.Throws<ObjectDisposedException>(fun () -> backend.SwiGlu(gate, disposed, destination)) |> ignore
    for g, u, d in [ half, up, destination; gate, half, destination; gate, up, half ] do
        Assert.Throws<ArgumentException>(fun () -> backend.SwiGlu(g, u, d)) |> ignore
    Assert.Equal<single>(Array.create 6 42.0f, download backend destination)

let private halfBits bits = BitConverter.UInt16BitsToHalf(bits)
let private bits (value: Half) = BitConverter.HalfToUInt16Bits(value)

let private uploadHalf (backend: ITensorBackend) (shape: TensorShape) (values: Half array) =
    backend.Upload(shape, ReadOnlyMemory<Half>(values))

let private downloadHalf (backend: ITensorBackend) (tensor: ITensor) =
    let values = Array.zeroCreate<Half> tensor.Shape.ElementCount
    backend.Download(tensor, values.AsMemory())
    values

// The F16 kernel computes in F32 and rounds once, about 2^-11 relative, plus Metal's exp error;
// an absolute floor covers subnormal results, which the GPU keeps exactly.
let private assertSwiGluHalfClose (expected: float) (actual: Half) =
    let tolerance = 1e-2 * abs expected + 1e-7
    Assert.True(abs (float actual - expected) <= tolerance, $"Expected {expected}, got {float actual}.")

[<MetalFact>]
[<Trait("Category", "Metal")>]
let ``Metal SwiGlu F16 matches a double-precision reference within F16 rounding`` () =
    use backend = new MetalBackend() :> ITensorBackend
    let random = Random(9012)
    let shape = TensorShape.Create [| 257 |]
    // Gates reach -20, beyond -11.1 where e^-x would overflow binary16: silu(-12) is about
    // -7.4e-5, a normal F16 that the F32 computation must keep instead of rounding to zero.
    let gateValues = Array.init 257 (fun _ -> Half.op_Explicit (float32 (random.NextDouble() * 50.0 - 20.0)))
    let upValues = Array.init 257 (fun _ -> Half.op_Explicit (float32 (random.NextDouble() * 20.0 - 10.0)))
    let expected = Array.map2 swiGluReference (Array.map float32 gateValues) (Array.map float32 upValues)
    use gate = uploadHalf backend shape gateValues
    use up = uploadHalf backend shape upValues
    use destination = backend.Allocate(shape, TensorEncoding.F16)

    backend.SwiGlu(gate, up, destination)
    Array.iter2 assertSwiGluHalfClose expected (downloadHalf backend destination)
    Assert.Equal<uint16>(Array.map bits gateValues, downloadHalf backend gate |> Array.map bits)
    Assert.Equal<uint16>(Array.map bits upValues, downloadHalf backend up |> Array.map bits)

    // Reuse either input as the destination, in F16 as well.
    backend.SwiGlu(gate, up, up)
    Array.iter2 assertSwiGluHalfClose expected (downloadHalf backend up)
    use upAgain = uploadHalf backend shape upValues
    backend.SwiGlu(gate, upAgain, gate)
    Array.iter2 assertSwiGluHalfClose expected (downloadHalf backend gate)

[<MetalFact>]
[<Trait("Category", "Metal")>]
let ``Metal SwiGlu F16 handles special gates and signed zero`` () =
    use backend = new MetalBackend() :> ITensorBackend
    let gateValues =
        [| Half.op_Explicit Single.PositiveInfinity; Half.op_Explicit Single.NegativeInfinity; Half.op_Explicit Single.NaN
           Half.op_Explicit 0.0f; Half.op_Explicit -0.0f; halfBits 0x0001us; halfBits 0x8001us; Half.op_Explicit 100.0f |]
    let shape = TensorShape.Create [| gateValues.Length |]
    use gate = uploadHalf backend shape gateValues
    use up = uploadHalf backend shape (Array.create gateValues.Length (Half.op_Explicit 2.0f))

    backend.SwiGlu(gate, up, up)

    let result = downloadHalf backend up
    Assert.Equal(0x7C00us, bits result[0]) // silu(+inf) = +inf, times 2
    Assert.True(Half.IsNaN result[1])     // silu(-inf) = -inf / inf = NaN
    Assert.True(Half.IsNaN result[2])     // NaN gate
    Assert.Equal(0x0000us, bits result[3]) // silu(0) = 0
    Assert.Equal(0x8000us, bits result[4]) // silu(-0) = -0, times 2
    // Subnormal gates: the exact result is about ±2^-24, and Metal does not guarantee how the
    // intermediate 2^-25 tie rounds, so accept either neighbour within the F16 tolerance.
    assertSwiGluHalfClose (swiGluReference (float32 gateValues[5]) 2.0f) result[5]
    assertSwiGluHalfClose (swiGluReference (float32 gateValues[6]) 2.0f) result[6]
    Assert.Equal(bits (Half.op_Explicit 200.0f), bits result[7]) // e^-100 vanishes next to 1, so silu(100) = 100
    Assert.Equal<uint16>(Array.map bits gateValues, downloadHalf backend gate |> Array.map bits)
