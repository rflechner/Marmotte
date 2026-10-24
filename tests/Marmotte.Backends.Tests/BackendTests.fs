module Marmotte.Backends.Tests

open System
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

/// Serves one Q4_0 matrix of two rows from memory: two 18-byte blocks per 64-value row.
type private MatrixSource() =
    let descriptor =
        { Name = "blk.0.attn_q.weight"
          Encoding = Some TensorEncoding.Q4_0
          SourceEncoding = "Q4_0"
          Shape = [| 2UL; 64UL |]
          ElementCount = 128UL
          ByteSize = Some 72UL }

    interface ITensorSource with
        member _.Tensors = [| descriptor |] :> IReadOnlyList<TensorDescriptor>
        member _.TryFindTensor(name) = if name = descriptor.Name then Some descriptor else None
        member _.ReadPayload(_) = ReadOnlyMemory<byte>(Array.zeroCreate 72)

    interface IDisposable with
        member _.Dispose() = ()

let private weightMatrix () =
    use source = new MatrixSource()
    WeightMatrix.Read(source, "blk.0.attn_q.weight")

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
    for left, right, destination in [ nullTensor, tensor, tensor; tensor, nullTensor, tensor; tensor, tensor, nullTensor ] do
        Assert.Throws<ArgumentNullException>(fun () -> backend.Add(left, right, destination)) |> ignore

[<Fact>]
let ``Add preserves ordinary floating point special values`` () =
    use backend = cpu ()
    let shape = TensorShape.Create [| 3 |]
    use left = backend.Upload(shape, ReadOnlyMemory<single>([| Single.NaN; Single.PositiveInfinity; Single.PositiveInfinity |]))
    use right = backend.Upload(shape, ReadOnlyMemory<single>([| 1.0f; 2.0f; Single.NegativeInfinity |]))
    backend.Add(left, right, left)
    let result = download backend left
    Assert.True(Single.IsNaN(result[0]))
    Assert.Equal(Single.PositiveInfinity, result[1])
    Assert.True(Single.IsNaN(result[2]))

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
    let operations =
        [ (fun () -> backend.Allocate(shape) |> ignore)
          (fun () -> backend.Upload(shape, ReadOnlyMemory<single>([| 1.0f |])) |> ignore)
          (fun () -> backend.UploadWeights(weightMatrix ()) |> ignore)
          (fun () -> backend.Download(tensor, Memory<single>([| 0.0f |])))
          (fun () -> backend.Synchronize())
          (fun () -> backend.Add(tensor, tensor, tensor)) ]

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
    backend.Dispose()
    backend.Dispose()

    Assert.Throws<ObjectDisposedException>(fun () -> backend.Allocate(shape) |> ignore) |> ignore
    Assert.Throws<ObjectDisposedException>(fun () -> backend.Upload(shape, ReadOnlyMemory<single>.Empty) |> ignore) |> ignore
    Assert.Throws<ObjectDisposedException>(fun () -> backend.UploadWeights(weightMatrix ()) |> ignore) |> ignore
    Assert.Throws<ObjectDisposedException>(fun () -> backend.Download(tensor, Memory<single>.Empty)) |> ignore
    Assert.Throws<ObjectDisposedException>(fun () -> backend.Synchronize()) |> ignore
    Assert.Throws<ObjectDisposedException>(fun () -> backend.Add(tensor, tensor, tensor)) |> ignore
