module Marmotte.Backends.HalfTests

open System
open Marmotte.Core
open Marmotte.Cpu
open Marmotte.Cuda
open Marmotte.Metal
open Marmotte.Backends.CudaTests
open Xunit

let private half bits = BitConverter.UInt16BitsToHalf(bits)
let private bits value = BitConverter.HalfToUInt16Bits(value)

let private download (backend: ITensorBackend) (tensor: ITensor) =
    let values = Array.zeroCreate<Half> tensor.Shape.ElementCount
    backend.Download(tensor, values.AsMemory())
    values

let private upload (backend: ITensorBackend) shape (values: Half array) =
    backend.Upload(shape, ReadOnlyMemory<Half>(values))

// Independent binary16 operands/results: exact sums, halfway rounding, overflow,
// the normal/subnormal boundary, signed zero, infinities and NaNs.
let private additionCases =
    [| 0x3c00us, 0x4000us, 0x4200us // 1 + 2 = 3
       0xbc00us, 0x4000us, 0x3c00us // -1 + 2 = 1
       0x3c00us, 0x1000us, 0x3c00us // 1 + 2^-11: tie rounds down to even
       0x3c01us, 0x1000us, 0x3c02us // odd significand: tie rounds up to even
       0xbc00us, 0x9000us, 0xbc00us
       0xbc01us, 0x9000us, 0xbc02us
       0x7bffus, 0x4800us, 0x7bffus // 65504 + 8 rounds to maximum finite
       0x7bffus, 0x4c00us, 0x7c00us // 65504 + 16 overflows at the midpoint
       0xfbffus, 0xcc00us, 0xfc00us
       0x0001us, 0x0001us, 0x0002us // Smallest subnormal doubled
       0x03ffus, 0x0001us, 0x0400us // Cross into normal values
       0x0400us, 0x8001us, 0x03ffus // Cross back into subnormals
       0x3c00us, 0xbc00us, 0x0000us // Exact cancellation
       0x8000us, 0x8000us, 0x8000us
       0x0000us, 0x8000us, 0x0000us
       0x7c00us, 0x3c00us, 0x7c00us
       0xfc00us, 0xbc00us, 0xfc00us
       0x7c00us, 0xfc00us, 0x7e00us
       0x7e35us, 0x0000us, 0x7e00us
       0x7d01us, 0x0000us, 0x7e00us |]

// Independent binary16 SwiGLU fixtures: gate, up and destination = silu(gate) × up, rounded once
// to nearest-even. Finite results were computed in double precision; each lies at least 0.08
// units in the last place from a rounding midpoint, so a few F32 ULP of exponential error
// cannot change the expected bits.
let private swiGluCases =
    [| 0x4000us, 0x4500us, 0x4867us // silu(2) × 5 ≈ 8.80797 rounds to 8.8046875
       0xc000us, 0x4200us, 0xb9b9us // silu(-2) × 3 ≈ -0.715218
       0x3c00us, 0x3c00us, 0x39d9us // silu(1) ≈ 0.731059
       0xbe00us, 0xc400us, 0x3c61us // silu(-1.5) × -4 ≈ 1.094553
       0x3400us, 0x4700us, 0x3bdfus // silu(0.25) × 7 ≈ 0.983809
       0x4200us, 0xc000us, 0xc5b7us // silu(3) × -2 ≈ -5.715445
       0x5640us, 0x3800us, 0x5240us // silu(100) × 0.5 = 50
       0xca00us, 0x3c00us, 0x84d5us // silu(-12) ≈ -7.373e-5, in the smallest normal binade
       0xcd00us, 0x3c00us, 0x8001us // silu(-20) ≈ -4.12e-8 rounds to the smallest subnormal
       0x0003us, 0x4000us, 0x0003us // A subnormal gate is not flushed to zero
       0x0000us, 0x4200us, 0x0000us
       0x8000us, 0x3c00us, 0x8000us // silu(-0) keeps its sign
       0xd640us, 0x3c00us, 0x8000us // e^100 overflows in F32: -100 / ∞ is a signed zero
       0x7bffus, 0x4000us, 0x7c00us // 65504 × 2 overflows binary16
       0x7bffus, 0xc000us, 0xfc00us
       0x7c00us, 0x3c00us, 0x7c00us // silu(+∞) = +∞
       0x7c00us, 0xc000us, 0xfc00us
       0xfc00us, 0x3c00us, 0x7e00us // silu(-∞) = -∞ / ∞ is NaN
       0x7e00us, 0x3c00us, 0x7e00us
       0x3c00us, 0x7e00us, 0x7e00us
       0x7c00us, 0x0000us, 0x7e00us |] // ∞ × 0 is NaN

let private assertResults (expected: uint16 array) (actual: Half array) =
    Assert.Equal(expected.Length, actual.Length)
    for index = 0 to expected.Length - 1 do
        if Half.IsNaN(half expected[index]) then
            Assert.True(Half.IsNaN actual[index], $"Expected NaN at {index}.")
        else
            Assert.Equal(expected[index], bits actual[index])

/// silu(gate) × up computed in double precision, independent of any device exponential,
/// then rounded once to binary16.
let private swiGluHalfReference (gate: Half) (up: Half) : Half =
    let x = float gate
    Half.op_Explicit (x / (1.0 + Math.Exp(-x)) * float up)

/// Position on an integer line where neighbouring binary16 values, including the largest
/// finite value and infinity, are one step apart. Both zeros map to 0.
let private halfOrdinal (value: Half) =
    let raw = int (bits value)
    if raw &&& 0x8000 = 0 then raw else -(raw &&& 0x7fff)

/// The device rounds an F32 result whose exponential carries a few ULP of error, so a value
/// within that error of a binary16 rounding midpoint may land on either neighbour.
let private assertWithinOneHalfUlp (expected: Half) (actual: Half) =
    if Half.IsNaN expected then
        Assert.True(Half.IsNaN actual, $"Expected NaN, got 0x{bits actual:x4}.")
    else
        let distance = abs (halfOrdinal expected - halfOrdinal actual)
        Assert.False(Half.IsNaN actual, $"Expected 0x{bits expected:x4}, got NaN.")
        Assert.True(distance <= 1, $"Expected 0x{bits expected:x4}, got 0x{bits actual:x4}.")

        // A zero result keeps the sign of the exact product.
        if halfOrdinal expected = 0 && halfOrdinal actual = 0 then
            Assert.Equal(bits expected, bits actual)

/// RMS normalization of each row in double precision, independent of the kernel's reduction
/// order, then rounded once to binary16.
let private rmsNormHalfReference (input: Half array) (columns: int) (weight: Half array) (epsilon: single) : Half array =
    let normalizeRow (row: Half array) =
        let mean = row |> Array.averageBy (fun value -> float value * float value)
        let scale = 1.0 / Math.Sqrt(mean + float epsilon)
        Array.map2 (fun value weightValue -> Half.op_Explicit(float value * scale * float weightValue)) row weight

    input |> Array.chunkBySize columns |> Array.collect normalizeRow

let private checkArithmetic (backend: ITensorBackend) =
    for dimensions in [ [||]; [| 0 |]; [| 2; 10 |]; [| 255 |]; [| 256 |]; [| 257 |]; [| 65537 |] ] do
        let shape = TensorShape.Create dimensions
        let cases = Array.init shape.ElementCount (fun i -> additionCases[i % additionCases.Length])
        let leftValues = cases |> Array.map (fun (left, _, _) -> half left)
        let rightValues = cases |> Array.map (fun (_, right, _) -> half right)
        let expected = cases |> Array.map (fun (_, _, result) -> result)
        use left = upload backend shape leftValues
        use right = upload backend shape rightValues
        use destination = backend.Allocate(shape, TensorEncoding.F16)
        Assert.Equal(TensorEncoding.F16, destination.Encoding)
        Assert.Equal(shape, destination.Shape)
        Assert.All(download backend destination, fun value -> Assert.Equal(0us, bits value))

        backend.Add(left, right, destination)
        assertResults expected (download backend destination)
        Assert.Equal<uint16>(Array.map bits leftValues, download backend left |> Array.map bits)
        Assert.Equal<uint16>(Array.map bits rightValues, download backend right |> Array.map bits)

        // Reuse either input; the other input is still the original operand.
        backend.Add(left, right, left)
        assertResults expected (download backend left)
        use originalLeft = upload backend shape leftValues
        backend.Add(originalLeft, right, right)
        assertResults expected (download backend right)

let private checkTransfersAndOrdering (backend: ITensorBackend) =
    // Every binary16 bit pattern, including signaling NaNs, must survive copying unchanged.
    let source = Array.init 65536 (uint16 >> half)
    let shape = TensorShape.Create [| source.Length |]
    use tensor = upload backend shape source
    Array.Fill(source, Half.Zero)
    let sentinel = half 0x3555us
    let destination = Array.create (shape.ElementCount + 2) sentinel
    backend.Download(tensor, destination.AsMemory(1, shape.ElementCount))
    Assert.Equal(bits sentinel, bits destination[0])
    Assert.Equal(bits sentinel, bits destination[destination.Length - 1])
    for index = 0 to shape.ElementCount - 1 do
        Assert.Equal(uint16 index, bits destination[index + 1])

    let input = [| sentinel; half 0x3c00us; half 0x4000us; sentinel |]
    let smallShape = TensorShape.Create [| 2 |]
    use small = backend.Upload(smallShape, ReadOnlyMemory<Half>(input, 1, 2))
    Array.Fill(input, Half.Zero)
    backend.Add(small, small, small)
    use output = backend.Allocate(smallShape, TensorEncoding.F16)
    backend.Add(small, small, output)
    // Disposing an input while its Add is pending must wait before releasing its storage.
    small.Dispose()
    use empty = backend.Allocate(TensorShape.Create [| 0 |], TensorEncoding.F16)
    backend.Download(empty, Memory<Half>.Empty)
    assertResults [| 0x4400us; 0x4800us |] (download backend output)

let private checkValidationAndLifetime (backend: ITensorBackend) (other: ITensorBackend) (shape: TensorShape) =
    let values = Array.create shape.ElementCount (half 0x3555us)
    use local = upload backend shape values
    use foreign = upload other shape values
    use f32 = backend.Allocate(shape)
    use wrongShape = backend.Allocate(TensorShape.Create [| 1; shape.ElementCount |], TensorEncoding.F16)
    let nullTensor = Unchecked.defaultof<ITensor>
    let nullShape = Unchecked.defaultof<TensorShape>
    let emptyShape = TensorShape.Create [| 0 |]

    for encoding in [ TensorEncoding.BF16; TensorEncoding.Q4_0; TensorEncoding.Q8_0; TensorEncoding.Q4_K; TensorEncoding.Q6_K ] do
        Assert.Throws<NotSupportedException>(fun () -> backend.Allocate(emptyShape, encoding) |> ignore) |> ignore
    Assert.Throws<ArgumentNullException>(fun () -> backend.Allocate(nullShape, TensorEncoding.F16) |> ignore) |> ignore
    Assert.Throws<ArgumentNullException>(fun () -> backend.Upload(nullShape, ReadOnlyMemory<Half>.Empty) |> ignore) |> ignore
    Assert.Throws<ArgumentException>(fun () -> backend.Upload(emptyShape, ReadOnlyMemory<Half>([| Half.One |])) |> ignore) |> ignore

    for invalid in [ foreign; f32; wrongShape ] do
        for left, right, destination in [ invalid, local, local; local, invalid, local; local, local, invalid ] do
            Assert.Throws<ArgumentException>(fun () -> backend.Add(left, right, destination)) |> ignore
    for left, right, destination in [ nullTensor, local, local; local, nullTensor, local; local, local, nullTensor ] do
        Assert.Throws<ArgumentNullException>(fun () -> backend.Add(left, right, destination)) |> ignore

    let hostHalf = Array.copy values
    let hostSingle = Array.create shape.ElementCount 42.0f
    Assert.Throws<ArgumentException>(fun () -> backend.Download(foreign, hostHalf.AsMemory())) |> ignore
    Assert.Throws<ArgumentException>(fun () -> backend.Download(f32, hostHalf.AsMemory())) |> ignore
    Assert.Throws<ArgumentException>(fun () -> backend.Download(local, hostSingle.AsMemory())) |> ignore
    Assert.Throws<ArgumentNullException>(fun () -> backend.Download(nullTensor, hostHalf.AsMemory())) |> ignore
    let wrongLength = Array.create (shape.ElementCount + 1) (half 0x3555us)
    Assert.Throws<ArgumentException>(fun () -> backend.Download(local, wrongLength.AsMemory())) |> ignore
    Assert.All(wrongLength, fun value -> Assert.Equal(0x3555us, bits value))
    Assert.Equal<Half>(values, hostHalf)
    Assert.All(hostSingle, fun value -> Assert.Equal(42.0f, value))
    Assert.Equal<Half>(values, download backend local)
    Assert.Equal<Half>(values, download other foreign)
    let checkSingle = Array.zeroCreate<single> shape.ElementCount
    backend.Download(f32, checkSingle.AsMemory())
    Assert.All(checkSingle, fun value -> Assert.Equal(0.0f, value))

    use disposed = backend.Allocate(shape, TensorEncoding.F16)
    disposed.Dispose()
    disposed.Dispose()
    for left, right, destination in [ disposed, local, local; local, disposed, local; local, local, disposed ] do
        Assert.Throws<ObjectDisposedException>(fun () -> backend.Add(left, right, destination)) |> ignore
    Assert.Throws<ObjectDisposedException>(fun () -> backend.Download(disposed, hostHalf.AsMemory())) |> ignore
    backend.Dispose()
    Assert.Equal(TensorEncoding.F16, local.Encoding)
    Assert.Equal(shape, local.Shape)
    Assert.Throws<ObjectDisposedException>(fun () -> backend.Allocate(shape, TensorEncoding.F16) |> ignore) |> ignore
    Assert.Throws<ObjectDisposedException>(fun () -> upload backend shape values |> ignore) |> ignore
    Assert.Throws<ObjectDisposedException>(fun () -> backend.Download(local, hostHalf.AsMemory())) |> ignore
    Assert.Throws<ObjectDisposedException>(fun () -> backend.Add(local, local, local)) |> ignore
    local.Dispose()
    local.Dispose()

[<Fact>]
let ``CPU F16 Add matches independent rounding and special value fixtures`` () =
    use backend = new CpuBackend() :> ITensorBackend
    checkArithmetic backend

[<Fact>]
let ``CPU F16 transfers preserve every bit pattern and support buffer reuse`` () =
    use backend = new CpuBackend() :> ITensorBackend
    checkTransfersAndOrdering backend

[<Fact>]
let ``CPU F16 validates encodings shapes ownership and disposal before writing`` () =
    use backend = new CpuBackend() :> ITensorBackend
    use other = new CpuBackend() :> ITensorBackend
    checkValidationAndLifetime backend other (TensorShape.Create [| 3 |])

[<Fact>]
let ``CUDA F16 empty tensors validate without loading the runtime`` () =
    use backend = new CudaBackend() :> ITensorBackend
    use other = new CudaBackend() :> ITensorBackend
    use empty = backend.Upload(TensorShape.Create [| 0 |], ReadOnlyMemory<Half>.Empty)
    backend.Add(empty, empty, empty)
    backend.SwiGlu(empty, empty, empty)
    backend.RmsNorm(empty, empty, 1e-5f, empty)
    backend.Download(empty, Memory<Half>.Empty)
    backend.Synchronize()
    checkValidationAndLifetime backend other (TensorShape.Create [| 0 |])

[<Fact>]
let ``CPU F32-only kernels reject F16 before writing`` () =
    use backend = new CpuBackend() :> ITensorBackend
    let shape = TensorShape.Create [| 1; 2 |]
    use halfTensor = upload backend shape [| half 0x3c00us; half 0x4000us |]
    use singleTensor = backend.Upload(shape, ReadOnlyMemory<single>([| 42.0f; 42.0f |]))
    for gate, up, destination in [ halfTensor, singleTensor, singleTensor; singleTensor, halfTensor, singleTensor; singleTensor, singleTensor, halfTensor ] do
        Assert.Throws<NotSupportedException>(fun () -> backend.SwiGlu(gate, up, destination)) |> ignore
    use weight = backend.Allocate(TensorShape.Create [| 2 |])
    Assert.Throws<NotSupportedException>(fun () -> backend.RmsNorm(halfTensor, weight, 1e-5f, singleTensor)) |> ignore
    let unchanged = Array.zeroCreate<single> 2
    backend.Download(singleTensor, unchanged.AsMemory())
    Assert.Equal<single>([| 42.0f; 42.0f |], unchanged)
    assertResults [| 0x3c00us; 0x4000us |] (download backend halfTensor)

[<Fact>]
let ``Metal Half operations remain explicit placeholders`` () =
    use backend = new MetalBackend() :> ITensorBackend
    let shape = TensorShape.Create [| 0 |]
    Assert.Throws<NotImplementedException>(fun () -> backend.Allocate(shape, TensorEncoding.F16) |> ignore) |> ignore
    Assert.Throws<NotImplementedException>(fun () -> backend.Upload(shape, ReadOnlyMemory<Half>.Empty) |> ignore) |> ignore
    Assert.Throws<NotImplementedException>(fun () -> backend.Download(Unchecked.defaultof<ITensor>, Memory<Half>.Empty)) |> ignore

[<CudaFact; Trait("Category", "CUDA")>]
let ``CUDA F16 Add matches independent rounding and special value fixtures`` () =
    use backend = new CudaBackend() :> ITensorBackend
    checkArithmetic backend

[<CudaFact; Trait("Category", "CUDA")>]
let ``CUDA F16 transfers preserve every bit pattern and order pending work`` () =
    use backend = new CudaBackend() :> ITensorBackend
    checkTransfersAndOrdering backend

[<CudaFact; Trait("Category", "CUDA")>]
let ``CUDA F16 validates encodings shapes ownership and disposal before writing`` () =
    use backend = new CudaBackend() :> ITensorBackend
    use other = new CudaBackend() :> ITensorBackend
    checkValidationAndLifetime backend other (TensorShape.Create [| 3 |])

[<CudaFact; Trait("Category", "CUDA")>]
let ``CUDA F16 SwiGlu matches independent rounding and special value fixtures`` () =
    use backend = new CudaBackend() :> ITensorBackend

    for dimensions in [ [||]; [| 0 |]; [| 2; 11 |]; [| 255 |]; [| 256 |]; [| 257 |]; [| 65537 |] ] do
        let shape = TensorShape.Create dimensions
        let cases = Array.init shape.ElementCount (fun i -> swiGluCases[i % swiGluCases.Length])
        let gateValues = cases |> Array.map (fun (gate, _, _) -> half gate)
        let upValues = cases |> Array.map (fun (_, up, _) -> half up)
        let expected = cases |> Array.map (fun (_, _, result) -> result)
        use gate = upload backend shape gateValues
        use up = upload backend shape upValues
        use destination = backend.Allocate(shape, TensorEncoding.F16)

        backend.SwiGlu(gate, up, destination)
        assertResults expected (download backend destination)
        Assert.Equal<uint16>(Array.map bits gateValues, download backend gate |> Array.map bits)
        Assert.Equal<uint16>(Array.map bits upValues, download backend up |> Array.map bits)

        // Either input may be the destination; the other input keeps its original values.
        backend.SwiGlu(gate, up, up)
        assertResults expected (download backend up)
        use originalUp = upload backend shape upValues
        backend.SwiGlu(gate, originalUp, gate)
        assertResults expected (download backend gate)

[<CudaFact; Trait("Category", "CUDA")>]
let ``CUDA F16 SwiGlu stays within one unit in the last place of a double-precision reference for every gate`` () =
    use backend = new CudaBackend() :> ITensorBackend
    // Every binary16 gate pattern, with up values that change the sign and scale the result
    // into the subnormal and overflow ranges.
    let upCycle = [| half 0x3c00us; half 0xc100us; half 0x3800us; half 0x7800us |] // 1, -2.5, 0.5, 32768
    let gateValues = Array.init 65536 (uint16 >> half)
    let upValues = Array.init gateValues.Length (fun index -> upCycle[index % upCycle.Length])
    let expected = Array.map2 swiGluHalfReference gateValues upValues
    let shape = TensorShape.Create [| gateValues.Length |]
    use gate = upload backend shape gateValues
    use up = upload backend shape upValues

    backend.SwiGlu(gate, up, up)
    Array.iter2 assertWithinOneHalfUlp expected (download backend up)

    // A queued F16 Add observes the SwiGlu result on the same stream:
    // silu(1) rounds to 0x39d9, and doubling it is exact.
    let smallShape = TensorShape.Create [| 3 |]
    use ones = upload backend smallShape (Array.create 3 (half 0x3c00us))
    use output = backend.Allocate(smallShape, TensorEncoding.F16)
    backend.SwiGlu(ones, ones, output)
    backend.Add(output, output, output)
    assertResults (Array.create 3 0x3dd9us) (download backend output)

[<CudaFact; Trait("Category", "CUDA")>]
let ``CUDA F16 RmsNorm matches independent rounding and special value fixtures`` () =
    use backend = new CudaBackend() :> ITensorBackend
    // Weight [1; 1]: a zero row stays +0, ±1 rows round to exactly ±1, a huge row normalizes
    // to 1, a NaN row is all NaN, and an infinite row scales to zero with NaN from inf × 0.
    let inputBits =
        [| 0x0000us; 0x0000us // 0, 0 -> +0, +0
           0x3c00us; 0xbc00us // 1, -1 -> +1, -1
           0x4000us; 0x4000us // 2, 2 -> 1, 1
           0x7bffus; 0x7bffus // 65504, 65504 -> 1, 1
           0x7e00us; 0x3c00us // NaN, 1 -> NaN, NaN
           0xfc00us; 0x3c00us |] // -inf, 1 -> NaN, +0
    let expectedBits =
        [| 0x0000us; 0x0000us
           0x3c00us; 0xbc00us
           0x3c00us; 0x3c00us
           0x3c00us; 0x3c00us
           0x7e00us; 0x7e00us
           0x7e00us; 0x0000us |]
    let shape = TensorShape.Create [| 6; 2 |]
    use weight = upload backend (TensorShape.Create [| 2 |]) [| half 0x3c00us; half 0x3c00us |]
    use input = upload backend shape (Array.map half inputBits)
    use destination = backend.Allocate(shape, TensorEncoding.F16)

    backend.RmsNorm(input, weight, 1e-5f, destination)
    assertResults expectedBits (download backend destination)
    Assert.Equal<uint16>(inputBits, download backend input |> Array.map bits)

    // Destination may be the input.
    backend.RmsNorm(input, weight, 1e-5f, input)
    assertResults expectedBits (download backend input)

    // Weight [-1; -1]: a zero row keeps the negative sign, and ±1 rows round to exactly ∓1.
    use minusOne = upload backend (TensorShape.Create [| 2 |]) [| half 0xbc00us; half 0xbc00us |]
    let zeroRow = TensorShape.Create [| 1; 2 |]
    use zeros = upload backend zeroRow [| Half.Zero; Half.Zero |]
    backend.RmsNorm(zeros, minusOne, 1e-5f, zeros)
    assertResults [| 0x8000us; 0x8000us |] (download backend zeros)
    use onesRow = upload backend zeroRow [| half 0x3c00us; half 0x3c00us |]
    backend.RmsNorm(onesRow, minusOne, 1e-5f, onesRow)
    assertResults [| 0xbc00us; 0xbc00us |] (download backend onesRow)

    // Weight [+inf; +inf]: a unit row overflows to +inf.
    use infinity = upload backend (TensorShape.Create [| 2 |]) [| half 0x7c00us; half 0x7c00us |]
    use unitRow = upload backend zeroRow [| half 0x3c00us; half 0x3c00us |]
    backend.RmsNorm(unitRow, infinity, 1e-5f, unitRow)
    assertResults [| 0x7c00us; 0x7c00us |] (download backend unitRow)

[<CudaFact; Trait("Category", "CUDA")>]
let ``CUDA F16 RmsNorm stays within one unit in the last place of a double-precision reference for every input pattern`` () =
    use backend = new CudaBackend() :> ITensorBackend
    // Every binary16 input pattern in 256 rows of 256 columns, with varied weights including
    // zeros, subnormals and infinities.
    let columns = 256
    let shape = TensorShape.Create [| 256; columns |]
    let inputValues = Array.init (256 * columns) (uint16 >> half)
    let weightValues = Array.init columns (fun index -> half (uint16 ((index * 257 + 32768) % 65536)))
    let expected = rmsNormHalfReference inputValues columns weightValues 1e-5f
    use input = upload backend shape inputValues
    use weight = upload backend (TensorShape.Create [| columns |]) weightValues

    backend.RmsNorm(input, weight, 1e-5f, input)
    Array.iter2 assertWithinOneHalfUlp expected (download backend input)
