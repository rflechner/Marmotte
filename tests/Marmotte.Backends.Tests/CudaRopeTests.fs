module Marmotte.Backends.CudaRopeTests

open System
open Marmotte.Core
open Marmotte.Cpu
open Marmotte.Cuda
open Marmotte.Backends.CudaTests
open Xunit

let private targetYarn =
    { ScalingFactor = 16.0
      OriginalContextLength = 16384
      BetaFast = 32.0
      BetaSlow = 1.0
      LogMultiplier = 1.0
      AttentionFactor = 1.0 }

let private download (backend: ITensorBackend) (tensor: ITensor) =
    let values = Array.zeroCreate<single> tensor.Shape.ElementCount
    backend.Download(tensor, values.AsMemory())
    values

let private cpuRotation (shape: TensorShape) (values: single array) positions embedding =
    use cpu = new CpuBackend() :> ITensorBackend
    use input = cpu.Upload(shape, ReadOnlyMemory<single>(values))
    cpu.Rope(input, ReadOnlyMemory<int>(positions), embedding, input)
    download cpu input

let private assertClose (expected: single) (actual: single) =
    if Single.IsNaN expected then
        Assert.True(Single.IsNaN actual, $"Expected NaN, got {actual}.")
    elif Single.IsInfinity expected then
        Assert.Equal(expected, actual)
    else
        let tolerance = 1e-6 * (1.0 + abs (float expected))
        Assert.True(abs (float expected - float actual) <= tolerance,
                    $"Expected {expected}, got {actual} (tolerance {tolerance}).")

[<Fact>]
let ``CUDA RoPE validates empty shapes positions and ownership without a driver`` () =
    use backend = new CudaBackend() :> ITensorBackend
    use other = new CudaBackend() :> ITensorBackend
    let embedding = RotaryEmbedding.Standard(8, 10000.0, RopePairing.AdjacentPairs)
    let emptyShape = TensorShape.Create [| 0; 8 |]
    use empty = backend.Allocate(emptyShape)
    use emptyHalf = backend.Allocate(emptyShape, TensorEncoding.F16)
    use foreign = other.Allocate(emptyShape)
    use partial = backend.Allocate(TensorShape.Create [| 0; 6 |])
    use badRank = backend.Allocate(TensorShape.Create [| 0 |])
    use different = backend.Allocate(TensorShape.Create [| 8; 0 |])
    let noPositions = ReadOnlyMemory<int>.Empty

    backend.Rope(empty, noPositions, embedding, empty)
    backend.Rope(emptyHalf, noPositions, embedding, emptyHalf)
    Assert.Empty(download backend empty)

    Assert.Throws<ArgumentException>(fun () -> backend.Rope(badRank, noPositions, embedding, badRank)) |> ignore
    Assert.Throws<ArgumentException>(fun () -> backend.Rope(partial, noPositions, embedding, partial)) |> ignore
    Assert.Throws<ArgumentException>(fun () -> backend.Rope(empty, ReadOnlyMemory<int>([| 0 |]), embedding, empty)) |> ignore
    Assert.Throws<ArgumentException>(fun () -> backend.Rope(empty, noPositions, embedding, different)) |> ignore
    Assert.Throws<ArgumentException>(fun () -> backend.Rope(empty, noPositions, embedding, emptyHalf)) |> ignore
    Assert.Throws<ArgumentException>(fun () -> backend.Rope(foreign, noPositions, embedding, empty)) |> ignore
    Assert.Throws<ArgumentNullException>(fun () -> backend.Rope(empty, noPositions, Unchecked.defaultof<RotaryEmbedding>, empty)) |> ignore
    Assert.Throws<ArgumentNullException>(fun () -> backend.Rope(Unchecked.defaultof<ITensor>, noPositions, embedding, empty)) |> ignore

    use zeroWidth = backend.Allocate(TensorShape.Create [| 1; 0 |])
    let negative = ReadOnlyMemory<int>([| -1 |])
    let error = Assert.Throws<ArgumentOutOfRangeException>(fun () -> backend.Rope(zeroWidth, negative, embedding, zeroWidth))
    Assert.Contains("index 0", error.Message)
    backend.Rope(zeroWidth, ReadOnlyMemory<int>([| 0 |]), embedding, zeroWidth)

    empty.Dispose()
    Assert.Throws<ObjectDisposedException>(fun () -> backend.Rope(empty, noPositions, embedding, empty)) |> ignore
    backend.Dispose()
    Assert.Throws<ObjectDisposedException>(fun () -> backend.Rope(emptyHalf, noPositions, embedding, emptyHalf)) |> ignore

[<CudaFact>]
[<Trait("Category", "CUDA")>]
let ``CUDA RoPE matches CPU for plain and YaRN angles with both pairings and head layouts`` () =
    use backend = new CudaBackend() :> ITensorBackend
    let positions = [| 0; 1; 16383; 262143 |]
    let random = Random(3024)

    for pairing in [ RopePairing.AdjacentPairs; RopePairing.SplitHalves ] do
        for embedding in [ RotaryEmbedding.Standard(128, 1000000.0, pairing)
                           RotaryEmbedding.Yarn(128, 1000000.0, pairing, targetYarn)
                           RotaryEmbedding.Yarn(128, 1000000.0, pairing,
                                                { targetYarn with LogMultiplier = 0.0; AttentionFactor = 1.5 }) ] do
            for heads in [ 8; 32 ] do
                for shape in [ TensorShape.Create [| positions.Length; heads * 128 |]
                               TensorShape.Create [| positions.Length; heads; 128 |] ] do
                    let values = Array.init shape.ElementCount (fun _ -> single (2.0 * random.NextDouble() - 1.0))
                    let expected = cpuRotation shape values positions embedding
                    use input = backend.Upload(shape, ReadOnlyMemory<single>(values))
                    use output = backend.Allocate(shape)

                    backend.Rope(input, ReadOnlyMemory<int>(positions), embedding, output)
                    Array.iter2 assertClose expected (download backend output)
                    Assert.Equal<single>(values, download backend input)

                    // Reuse coefficient staging and verify aliased input/output.
                    backend.Rope(input, ReadOnlyMemory<int>(positions), embedding, input)
                    Array.iter2 assertClose expected (download backend input)

[<CudaFact>]
[<Trait("Category", "CUDA")>]
let ``CUDA RoPE preserves stream ordering and copies positions before returning`` () =
    use backend = new CudaBackend() :> ITensorBackend
    let shape = TensorShape.Create [| 2; 8 |]
    let values = Array.init 16 (fun i -> single (i - 8) / 5.0f)
    let embedding = RotaryEmbedding.Standard(8, 10000.0, RopePairing.AdjacentPairs)
    let firstPositions = [| 1; 7 |]
    let secondPositions = [| 0; 16384 |]
    let afterFirst = cpuRotation shape values firstPositions embedding
    let expected = cpuRotation shape afterFirst secondPositions embedding
    use input = backend.Upload(shape, ReadOnlyMemory<single>(values))
    use output = backend.Allocate(shape)

    backend.Rope(input, ReadOnlyMemory<int>(firstPositions), embedding, output)
    firstPositions[0] <- 1000
    firstPositions[1] <- 2000
    backend.Rope(output, ReadOnlyMemory<int>(secondPositions), embedding, output)
    secondPositions[0] <- 1000
    secondPositions[1] <- 2000
    Array.iter2 assertClose expected (download backend output)

[<CudaFact>]
[<Trait("Category", "CUDA")>]
let ``CUDA RoPE F16 widens before arithmetic and rounds output once`` () =
    use backend = new CudaBackend() :> ITensorBackend
    let shape = TensorShape.Create [| 3; 2; 128 |]
    let positions = [| 0; 16384; 262143 |]
    let embedding = RotaryEmbedding.Yarn(128, 1000000.0, RopePairing.SplitHalves,
                                         { targetYarn with LogMultiplier = 0.0; AttentionFactor = 1.5 })
    let random = Random(4017)
    let values = Array.init shape.ElementCount (fun _ ->
        Half.op_Explicit(2.0 * random.NextDouble() - 1.0))
    let widened = values |> Array.map (fun value -> single (float value))
    let expected = cpuRotation shape widened positions embedding
    use input = backend.Upload(shape, ReadOnlyMemory<Half>(values))
    use destination = backend.Allocate(shape, TensorEncoding.F16)

    let readHalf (tensor: ITensor) =
        let output = Array.zeroCreate<Half> values.Length
        backend.Download(tensor, output.AsMemory())
        output

    backend.Rope(input, ReadOnlyMemory<int>(positions), embedding, destination)
    let actual = readHalf destination
    for i = 0 to values.Length - 1 do
        let expectedHalf = Half.op_Explicit(float expected[i])
        let tolerance = 0.001 * (1.0 + abs (float expectedHalf))
        Assert.True(abs (float expectedHalf - float actual[i]) <= tolerance,
                    $"F16 index {i}: expected {expectedHalf}, got {actual[i]}.")
    Assert.Equal<Half>(values, readHalf input)

    backend.Rope(input, ReadOnlyMemory<int>(positions), embedding, input)
    Assert.Equal<Half>(actual, readHalf input)

[<CudaFact>]
[<Trait("Category", "CUDA")>]
let ``CUDA RoPE rejects invalid tensors before touching destination`` () =
    use backend = new CudaBackend() :> ITensorBackend
    use other = new CudaBackend() :> ITensorBackend
    let embedding = RotaryEmbedding.Standard(4, 10000.0, RopePairing.AdjacentPairs)
    let shape = TensorShape.Create [| 2; 8 |]
    use input = backend.Upload(shape, ReadOnlyMemory<single>(Array.create 16 1.0f))
    use output = backend.Upload(shape, ReadOnlyMemory<single>(Array.create 16 42.0f))
    use half = backend.Upload(shape, ReadOnlyMemory<Half>(Array.create 16 Half.One))
    use foreign = other.Allocate(shape)
    use partial = backend.Allocate(TensorShape.Create [| 2; 6 |])
    use wrongShape = backend.Allocate(TensorShape.Create [| 2; 4 |])
    use disposed = backend.Allocate(shape)
    disposed.Dispose()
    let positions = ReadOnlyMemory<int>([| 0; 1 |])

    Assert.Throws<ArgumentException>(fun () -> backend.Rope(partial, positions, embedding, partial)) |> ignore
    Assert.Throws<ArgumentException>(fun () -> backend.Rope(input, positions, embedding, wrongShape)) |> ignore
    Assert.Throws<ArgumentException>(fun () -> backend.Rope(input, positions, embedding, half)) |> ignore
    Assert.Throws<ArgumentException>(fun () -> backend.Rope(foreign, positions, embedding, output)) |> ignore
    Assert.Throws<ObjectDisposedException>(fun () -> backend.Rope(disposed, positions, embedding, output)) |> ignore
    let error = Assert.Throws<ArgumentOutOfRangeException>(fun () ->
        backend.Rope(input, ReadOnlyMemory<int>([| 0; -1 |]), embedding, output))
    Assert.Contains("index 1", error.Message)
    Assert.Equal<single>(Array.create 16 42.0f, download backend output)
