module Marmotte.Backends.CudaTests

open System
open Marmotte.Core
open Marmotte.Cpu
open Marmotte.Cuda
open Xunit

/// Explicit opt-in: when enabled, missing libraries, drivers or devices fail the tests.
type CudaFactAttribute() as this =
    inherit FactAttribute()
    do
        if Environment.GetEnvironmentVariable("MARMOTTE_TEST_CUDA") <> "1" then
            this.Skip <- "Set MARMOTTE_TEST_CUDA=1 and deploy the native library; see docs/cuda.md."

let private download (backend: ITensorBackend) (tensor: ITensor) =
    let values = Array.zeroCreate<single> tensor.Shape.ElementCount
    backend.Download(tensor, values.AsMemory())
    values

[<Fact>]
let ``CUDA empty tensors need no runtime and still enforce shape ownership and lifetime`` () =
    use backend = new CudaBackend() :> ITensorBackend
    use otherBackend = new CudaBackend() :> ITensorBackend
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
    Assert.Equal(BackendKind.Cuda, destination.Backend)
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
let ``CUDA validates arguments before native loading or host writes`` () =
    use backend = new CudaBackend() :> ITensorBackend
    let nullShape = Unchecked.defaultof<TensorShape>
    let nullTensor = Unchecked.defaultof<ITensor>
    let emptyShape = TensorShape.Create [| 0 |]
    use empty = backend.Allocate(emptyShape)
    Assert.Throws<ArgumentException>(fun () -> new CudaBackend(-1) |> ignore) |> ignore
    Assert.Throws<ArgumentNullException>(fun () -> backend.Allocate(nullShape) |> ignore) |> ignore
    Assert.Throws<ArgumentNullException>(fun () -> backend.Upload(nullShape, ReadOnlyMemory<single>.Empty) |> ignore) |> ignore
    Assert.Throws<ArgumentException>(fun () -> backend.Upload(TensorShape.Create [| 1 |], ReadOnlyMemory<single>.Empty) |> ignore) |> ignore

    for a, b, output in [ nullTensor, empty, empty; empty, nullTensor, empty; empty, empty, nullTensor ] do
        Assert.Throws<ArgumentNullException>(fun () -> backend.Add(a, b, output)) |> ignore

    let sentinel = [| 42.0f |]
    Assert.Throws<ArgumentNullException>(fun () -> backend.Download(nullTensor, sentinel.AsMemory())) |> ignore
    Assert.Throws<ArgumentException>(fun () -> backend.Download(empty, sentinel.AsMemory())) |> ignore
    Assert.Equal<single>([| 42.0f |], sentinel)

[<CudaFact>]
[<Trait("Category", "CUDA")>]
let ``CUDA Add matches CPU for scalars matrices block boundaries and special values`` () =
    use backend = new CudaBackend() :> ITensorBackend
    use cpu = new CpuBackend() :> ITensorBackend
    let shapes = [ [||]; [| 0 |]; [| 2; 3 |]; [| 255 |]; [| 256 |]; [| 257 |]; [| 65537 |] ]

    for dimensions in shapes do
        let shape = TensorShape.Create dimensions
        let leftValues = Array.init shape.ElementCount (fun i -> single (i % 127 - 63) / 8.0f)
        let rightValues = Array.init shape.ElementCount (fun i -> single (i % 17) / 4.0f)
        use left = backend.Upload(shape, ReadOnlyMemory<single>(leftValues))
        use right = backend.Upload(shape, ReadOnlyMemory<single>(rightValues))
        use output = backend.Allocate(shape)
        Assert.All(download backend output, fun value -> Assert.Equal(0.0f, value))
        use cpuLeft = cpu.Upload(shape, ReadOnlyMemory<single>(leftValues))
        use cpuRight = cpu.Upload(shape, ReadOnlyMemory<single>(rightValues))
        use cpuOutput = cpu.Allocate(shape)

        backend.Add(left, right, output)
        cpu.Add(cpuLeft, cpuRight, cpuOutput)
        Assert.Equal<single>(download cpu cpuOutput, download backend output)
        Assert.Equal<single>(leftValues, download backend left)
        Assert.Equal<single>(rightValues, download backend right)

    let leftSpecial = [| Single.NaN; Single.PositiveInfinity; Single.PositiveInfinity; -0.0f; Single.Epsilon |]
    let rightSpecial = [| 1.0f; 2.0f; Single.NegativeInfinity; -0.0f; Single.Epsilon |]
    let shape = TensorShape.Create [| leftSpecial.Length |]
    use left = backend.Upload(shape, ReadOnlyMemory<single>(leftSpecial))
    use right = backend.Upload(shape, ReadOnlyMemory<single>(rightSpecial))
    backend.Add(left, right, left)
    let actual = download backend left
    Assert.True(Single.IsNaN(actual[0]))
    Assert.Equal(Single.PositiveInfinity, actual[1])
    Assert.True(Single.IsNaN(actual[2]))
    Assert.Equal(BitConverter.SingleToInt32Bits(-0.0f), BitConverter.SingleToInt32Bits(actual[3]))
    Assert.Equal(2.0f * Single.Epsilon, actual[4])

[<CudaFact>]
[<Trait("Category", "CUDA")>]
let ``CUDA queues dependent in-place operations and owns transfer buffers`` () =
    use backend = new CudaBackend() :> ITensorBackend
    let shape = TensorShape.Create [| 257 |]
    let host = Array.create 259 2.0f
    use left = backend.Upload(shape, ReadOnlyMemory<single>(host, 1, 257))
    Array.Fill(host, 99.0f)
    use right = backend.Upload(shape, ReadOnlyMemory<single>(Array.create 257 3.0f))
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

[<CudaFact>]
[<Trait("Category", "CUDA")>]
let ``CUDA rejects invalid device tensors before changing destination`` () =
    use backend = new CudaBackend() :> ITensorBackend
    use otherBackend = new CudaBackend() :> ITensorBackend
    use cpu = new CpuBackend() :> ITensorBackend
    let shape = TensorShape.Create [| 2; 3 |]
    use left = backend.Upload(shape, ReadOnlyMemory<single>(Array.create 6 2.0f))
    use destination = backend.Upload(shape, ReadOnlyMemory<single>(Array.create 6 42.0f))
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

[<CudaFact>]
[<Trait("Category", "CUDA")>]
let ``CUDA disposal waits for queued work and releases only its own resources`` () =
    use backend = new CudaBackend() :> ITensorBackend
    use otherBackend = new CudaBackend() :> ITensorBackend
    let shape = TensorShape.Create [| 65537 |]
    use input = backend.Upload(shape, ReadOnlyMemory<single>(Array.create shape.ElementCount 3.0f))
    use output = backend.Allocate(shape)
    use survivor = otherBackend.Upload(shape, ReadOnlyMemory<single>(Array.create shape.ElementCount 7.0f))
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
