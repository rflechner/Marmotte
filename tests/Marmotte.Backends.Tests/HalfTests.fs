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

let private assertResults (expected: uint16 array) (actual: Half array) =
    Assert.Equal(expected.Length, actual.Length)
    for index = 0 to expected.Length - 1 do
        if Half.IsNaN(half expected[index]) then
            Assert.True(Half.IsNaN actual[index], $"Expected NaN at {index}.")
        else
            Assert.Equal(expected[index], bits actual[index])

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
