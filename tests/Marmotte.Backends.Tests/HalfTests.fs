module Marmotte.Backends.HalfTests

open System
open Marmotte.Core
open Marmotte.Cpu
open Marmotte.Cuda
open Marmotte.Metal
open Marmotte.Backends.MetalTests
open Xunit

let private half bits = BitConverter.UInt16BitsToHalf(bits)
let private bits value = BitConverter.HalfToUInt16Bits(value)

let private download (backend: ITensorBackend) (tensor: ITensor) =
    let values = Array.zeroCreate<Half> tensor.Shape.ElementCount
    backend.Download(tensor, values.AsMemory())
    values

let private upload (backend: ITensorBackend) shape (values: Half array) =
    backend.Upload(shape, ReadOnlyMemory<Half>(values))

// F16 literals are written sign_exponent_mantissa, the fields of binary16.
// Independent binary16 operands/results: exact sums, halfway rounding, overflow,
// the normal/subnormal boundary, signed zero, infinities and NaNs.
let private additionCases =
    [| 0b0_01111_0000000000us, 0b0_10000_0000000000us, 0b0_10000_1000000000us // 1 + 2 = 3
       0b1_01111_0000000000us, 0b0_10000_0000000000us, 0b0_01111_0000000000us // -1 + 2 = 1
       0b0_01111_0000000000us, 0b0_00100_0000000000us, 0b0_01111_0000000000us // 1 + 2^-11: tie rounds down to even
       0b0_01111_0000000001us, 0b0_00100_0000000000us, 0b0_01111_0000000010us // odd significand: tie rounds up to even
       0b1_01111_0000000000us, 0b1_00100_0000000000us, 0b1_01111_0000000000us
       0b1_01111_0000000001us, 0b1_00100_0000000000us, 0b1_01111_0000000010us
       0b0_11110_1111111111us, 0b0_10010_0000000000us, 0b0_11110_1111111111us // 65504 + 8 rounds to maximum finite
       0b0_11110_1111111111us, 0b0_10011_0000000000us, 0b0_11111_0000000000us // 65504 + 16 overflows at the midpoint
       0b1_11110_1111111111us, 0b1_10011_0000000000us, 0b1_11111_0000000000us
       0b0_00000_0000000001us, 0b0_00000_0000000001us, 0b0_00000_0000000010us // Smallest subnormal doubled
       0b0_00000_1111111111us, 0b0_00000_0000000001us, 0b0_00001_0000000000us // Cross into normal values
       0b0_00001_0000000000us, 0b1_00000_0000000001us, 0b0_00000_1111111111us // Cross back into subnormals
       0b0_01111_0000000000us, 0b1_01111_0000000000us, 0b0_00000_0000000000us // Exact cancellation
       0b1_00000_0000000000us, 0b1_00000_0000000000us, 0b1_00000_0000000000us
       0b0_00000_0000000000us, 0b1_00000_0000000000us, 0b0_00000_0000000000us
       0b0_11111_0000000000us, 0b0_01111_0000000000us, 0b0_11111_0000000000us
       0b1_11111_0000000000us, 0b1_01111_0000000000us, 0b1_11111_0000000000us
       0b0_11111_0000000000us, 0b1_11111_0000000000us, 0b0_11111_1000000000us
       0b0_11111_1000110101us, 0b0_00000_0000000000us, 0b0_11111_1000000000us
       0b0_11111_0100000001us, 0b0_00000_0000000000us, 0b0_11111_1000000000us |]

let private assertResults (expected: uint16 array) (actual: Half array) =
    Assert.Equal(expected.Length, actual.Length)
    for index = 0 to expected.Length - 1 do
        if Half.IsNaN(half expected[index]) then
            Assert.True(Half.IsNaN actual[index], $"Expected NaN at {index}.")
        else
            Assert.Equal(expected[index], bits actual[index])

let private checkArithmetic (backend: ITensorBackend) =
    for dimensions in [ [||]; [| 0 |]; [| 2; 10 |]; [| 31 |]; [| 32 |]; [| 33 |]; [| 255 |]; [| 256 |]; [| 257 |]; [| 65537 |] ] do
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
    let sentinel = half 0b0_01101_0101010101us
    let destination = Array.create (shape.ElementCount + 2) sentinel
    backend.Download(tensor, destination.AsMemory(1, shape.ElementCount))
    Assert.Equal(bits sentinel, bits destination[0])
    Assert.Equal(bits sentinel, bits destination[destination.Length - 1])
    for index = 0 to shape.ElementCount - 1 do
        Assert.Equal(uint16 index, bits destination[index + 1])

    let input = [| sentinel; half 0b0_01111_0000000000us; half 0b0_10000_0000000000us; sentinel |]
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
    assertResults [| 0b0_10001_0000000000us; 0b0_10010_0000000000us |] (download backend output)

let private checkValidationAndLifetime (backend: ITensorBackend) (other: ITensorBackend) (shape: TensorShape) =
    let values = Array.create shape.ElementCount (half 0b0_01101_0101010101us)
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
    let wrongLength = Array.create (shape.ElementCount + 1) (half 0b0_01101_0101010101us)
    Assert.Throws<ArgumentException>(fun () -> backend.Download(local, wrongLength.AsMemory())) |> ignore
    Assert.All(wrongLength, fun value -> Assert.Equal(0b0_01101_0101010101us, bits value))
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
let ``CPU F32-only kernels reject F16 before writing`` () =
    use backend = new CpuBackend() :> ITensorBackend
    let shape = TensorShape.Create [| 1; 2 |]
    use halfTensor = upload backend shape [| half 0b0_01111_0000000000us; half 0b0_10000_0000000000us |]
    use singleTensor = backend.Upload(shape, ReadOnlyMemory<single>([| 42.0f; 42.0f |]))
    for gate, up, destination in [ halfTensor, singleTensor, singleTensor; singleTensor, halfTensor, singleTensor; singleTensor, singleTensor, halfTensor ] do
        Assert.Throws<NotSupportedException>(fun () -> backend.SwiGlu(gate, up, destination)) |> ignore
    use weight = backend.Allocate(TensorShape.Create [| 2 |])
    Assert.Throws<NotSupportedException>(fun () -> backend.RmsNorm(halfTensor, weight, 1e-5f, singleTensor)) |> ignore
    let unchanged = Array.zeroCreate<single> 2
    backend.Download(singleTensor, unchanged.AsMemory())
    Assert.Equal<single>([| 42.0f; 42.0f |], unchanged)
    assertResults [| 0b0_01111_0000000000us; 0b0_10000_0000000000us |] (download backend halfTensor)

// Independent binary32 + binary16 operands and binary32 results: F32 rounding that F16 arithmetic
// could not produce, F16 subnormals that widen to normal F32, sums beyond the F16 range,
// signed zero, infinities and NaNs.
let private mixedAdditionCases =
    [| 0x3FC00000u, 0b0_01111_0000000000us, 0x40200000u // 1.5 + 1 = 2.5
       0xBFC00000u, 0b1_01111_0000000000us, 0xC0200000u
       0x3F800001u, 0b0_00100_0000000000us, 0x3F801001u // (1 + 2^-23) + 2^-11 is exact in F32
       0x3F800003u, 0b0_01111_0000000000us, 0x40000002u // Tie in F32 rounds up to even; an F16 sum would give 2
       0x00000000u, 0b0_00000_0000000001us, 0x33800000u // Smallest F16 subnormal is a normal F32
       0x00000000u, 0b0_00000_1111111111us, 0x387FC000u // Largest F16 subnormal
       0x477FE000u, 0b0_11110_1111111111us, 0x47FFE000u // 65504 + 65504 stays finite in F32
       0xBF800000u, 0b0_01111_0000000000us, 0x00000000u // Exact cancellation
       0x80000000u, 0b1_00000_0000000000us, 0x80000000u
       0x80000000u, 0b0_00000_0000000000us, 0x00000000u
       0x3F800000u, 0b0_11111_0000000000us, 0x7F800000u
       0x3F800000u, 0b1_11111_0000000000us, 0xFF800000u
       0x7F800000u, 0b1_11111_0000000000us, 0x7FC00000u
       0x3F800000u, 0b0_11111_1000000000us, 0x7FC00000u
       0x7FC00000u, 0b0_01111_0000000000us, 0x7FC00000u |]

let private single' (bits: uint32) = BitConverter.UInt32BitsToSingle(bits)
let private singleBits (value: single) = BitConverter.SingleToUInt32Bits(value)

let private downloadSingle (backend: ITensorBackend) (tensor: ITensor) =
    let values = Array.zeroCreate<single> tensor.Shape.ElementCount
    backend.Download(tensor, values.AsMemory())
    values

let private uploadSingle (backend: ITensorBackend) shape (values: single array) =
    backend.Upload(shape, ReadOnlyMemory<single>(values))

let private assertSingleResults (expected: uint32 array) (actual: single array) =
    Assert.Equal(expected.Length, actual.Length)
    for index = 0 to expected.Length - 1 do
        if Single.IsNaN(single' expected[index]) then
            Assert.True(Single.IsNaN actual[index], $"Expected NaN at {index}.")
        else
            Assert.Equal(expected[index], singleBits actual[index])

let private checkMixedArithmetic (backend: ITensorBackend) =
    for dimensions in [ [||]; [| 0 |]; [| 3; 5 |]; [| 31 |]; [| 32 |]; [| 33 |]; [| 257 |]; [| 65537 |] ] do
        let shape = TensorShape.Create dimensions
        let cases = Array.init shape.ElementCount (fun i -> mixedAdditionCases[i % mixedAdditionCases.Length])
        let leftValues = cases |> Array.map (fun (left, _, _) -> single' left)
        let rightValues = cases |> Array.map (fun (_, right, _) -> half right)
        let expected = cases |> Array.map (fun (_, _, result) -> result)
        use left = uploadSingle backend shape leftValues
        use right = upload backend shape rightValues
        use destination = backend.Allocate(shape)

        backend.AddF32F16(left, right, destination)
        assertSingleResults expected (downloadSingle backend destination)
        Assert.Equal<uint32>(Array.map singleBits leftValues, downloadSingle backend left |> Array.map singleBits)
        Assert.Equal<uint16>(Array.map bits rightValues, download backend right |> Array.map bits)

        // The residual stream is usually updated in place.
        backend.AddF32F16(left, right, left)
        assertSingleResults expected (downloadSingle backend left)

let private checkMixedValidation (backend: ITensorBackend) (other: ITensorBackend) (shape: TensorShape) =
    let count = shape.ElementCount
    use f32 = uploadSingle backend shape (Array.create count 42.0f)
    use f16 = upload backend shape (Array.create count (half 0b0_01111_0000000000us))
    use foreignF32 = uploadSingle other shape (Array.create count 1.0f)
    use foreignF16 = upload other shape (Array.create count (half 0b0_01111_0000000000us))
    use wrongShape = backend.Allocate(TensorShape.Create [| 1; count |])
    let nullTensor = Unchecked.defaultof<ITensor>

    let incompatible =
        [ f16, f16, f32
          f32, f32, f32
          f32, f16, f16
          f16, f32, f32
          foreignF32, f16, f32
          f32, foreignF16, f32
          f32, f16, foreignF32
          wrongShape, f16, f32
          f32, f16, wrongShape ]

    for left, right, destination in incompatible do
        Assert.Throws<ArgumentException>(fun () -> backend.AddF32F16(left, right, destination)) |> ignore

    for left, right, destination in [ nullTensor, f16, f32; f32, nullTensor, f32; f32, f16, nullTensor ] do
        Assert.Throws<ArgumentNullException>(fun () -> backend.AddF32F16(left, right, destination)) |> ignore

    use disposed = backend.Allocate(shape, TensorEncoding.F16)
    disposed.Dispose()
    Assert.Throws<ObjectDisposedException>(fun () -> backend.AddF32F16(f32, disposed, f32)) |> ignore
    Assert.All(downloadSingle backend f32, fun value -> Assert.Equal(42.0f, value))

[<Fact>]
let ``CPU AddF32F16 matches independent rounding and special value fixtures`` () =
    use backend = new CpuBackend() :> ITensorBackend
    checkMixedArithmetic backend

[<Fact>]
let ``CPU AddF32F16 validates encodings shapes ownership and disposal before writing`` () =
    use backend = new CpuBackend() :> ITensorBackend
    use other = new CpuBackend() :> ITensorBackend
    checkMixedValidation backend other (TensorShape.Create [| 3 |])

[<Fact>]
let ``CUDA Half operations remain explicit placeholders`` () =
    use backend = new CudaBackend() :> ITensorBackend
    let shape = TensorShape.Create [| 0 |]
    Assert.Throws<NotImplementedException>(fun () -> backend.Allocate(shape, TensorEncoding.F16) |> ignore) |> ignore
    Assert.Throws<NotImplementedException>(fun () -> backend.Upload(shape, ReadOnlyMemory<Half>.Empty) |> ignore) |> ignore
    Assert.Throws<NotImplementedException>(fun () -> backend.Download(Unchecked.defaultof<ITensor>, Memory<Half>.Empty)) |> ignore

[<Fact>]
let ``Metal F16 empty tensors validate without loading the runtime`` () =
    use backend = new MetalBackend() :> ITensorBackend
    use other = new MetalBackend() :> ITensorBackend
    use empty = backend.Upload(TensorShape.Create [| 0 |], ReadOnlyMemory<Half>.Empty)
    backend.Add(empty, empty, empty)
    backend.Download(empty, Memory<Half>.Empty)
    backend.Synchronize()
    checkValidationAndLifetime backend other (TensorShape.Create [| 0 |])

[<Fact>]
let ``Metal AddF32F16 empty tensors validate without loading the runtime`` () =
    use backend = new MetalBackend() :> ITensorBackend
    use other = new MetalBackend() :> ITensorBackend
    checkMixedValidation backend other (TensorShape.Create [| 0 |])

[<MetalFact; Trait("Category", "Metal")>]
let ``Metal F16 Add matches independent rounding and special value fixtures`` () =
    use backend = new MetalBackend() :> ITensorBackend
    checkArithmetic backend

[<MetalFact; Trait("Category", "Metal")>]
let ``Metal F16 transfers preserve every bit pattern and order pending work`` () =
    use backend = new MetalBackend() :> ITensorBackend
    checkTransfersAndOrdering backend

[<MetalFact; Trait("Category", "Metal")>]
let ``Metal F16 validates encodings shapes ownership and disposal before writing`` () =
    use backend = new MetalBackend() :> ITensorBackend
    use other = new MetalBackend() :> ITensorBackend
    checkValidationAndLifetime backend other (TensorShape.Create [| 3 |])

[<MetalFact; Trait("Category", "Metal")>]
let ``Metal AddF32F16 matches independent rounding and special value fixtures`` () =
    use backend = new MetalBackend() :> ITensorBackend
    checkMixedArithmetic backend

[<MetalFact; Trait("Category", "Metal")>]
let ``Metal AddF32F16 validates encodings shapes ownership and disposal before writing`` () =
    use backend = new MetalBackend() :> ITensorBackend
    use other = new MetalBackend() :> ITensorBackend
    checkMixedValidation backend other (TensorShape.Create [| 3 |])
