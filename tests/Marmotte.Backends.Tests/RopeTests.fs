module Marmotte.Backends.RopeTests

open System
open Marmotte.Core
open Marmotte.Cpu
open Xunit

let private cpu () = new CpuBackend() :> ITensorBackend

let private download (backend: ITensorBackend) (tensor: ITensor) =
    let values = Array.zeroCreate<single> tensor.Shape.ElementCount
    backend.Download(tensor, values.AsMemory())
    values

let private rope (backend: ITensorBackend) (dimensions: int array) (values: single array) (positions: int array) embedding =
    use input = backend.Upload(TensorShape.Create dimensions, ReadOnlyMemory<single>(values))
    use destination = backend.Allocate(TensorShape.Create dimensions)
    backend.Rope(input, ReadOnlyMemory<int>(positions), embedding, destination)
    Assert.Equal<single>(values, download backend input)
    download backend destination

/// Rotation in double precision from the RoFormer formulas, with frequencies computed independently
/// of RotaryEmbedding as exp(-2j / d × ln base). Returns each value with the magnitude of its terms.
let private ropeReference (values: single array) (positions: int array) headDimension (frequencyBase: float) pairing =
    let pairs = headDimension / 2
    let valuesPerToken = values.Length / positions.Length
    let expected = Array.zeroCreate<float> values.Length
    let scale = Array.zeroCreate<float> values.Length
    for token in 0 .. positions.Length - 1 do
        for head in 0 .. valuesPerToken / headDimension - 1 do
            for pair in 0 .. pairs - 1 do
                let angle = float positions[token] * Math.Exp(-2.0 * float pair / float headDimension * Math.Log frequencyBase)
                let start = token * valuesPerToken + head * headDimension
                let first, second =
                    match pairing with
                    | RopePairing.AdjacentPairs -> start + 2 * pair, start + 2 * pair + 1
                    | RopePairing.SplitHalves -> start + pair, start + pair + pairs
                let a = float values[first]
                let b = float values[second]
                expected[first] <- a * Math.Cos angle - b * Math.Sin angle
                expected[second] <- a * Math.Sin angle + b * Math.Cos angle
                scale[first] <- abs a + abs b
                scale[second] <- abs a + abs b
    expected, scale

let private assertMatchesReference (actual: single array) (expected: float array, scale: float array) =
    for index in 0 .. actual.Length - 1 do
        let tolerance = 1e-6 * scale[index] + 1e-30
        Assert.True(abs (float actual[index] - expected[index]) <= tolerance, $"Value {index}: expected {expected[index]}, got {actual[index]}.")

let private randomValues (random: Random) count =
    Array.init count (fun _ -> float32 (random.NextDouble() * 2.0 - 1.0))

[<Fact>]
let ``Rope rotates hand-computed pairs at their frequencies`` () =
    use backend = cpu ()
    // Head of 4 values, base 10000: pair 0 turns by 1 radian per position and pair 1 by 0.01.
    let embedding = RotaryEmbedding.Standard(4, 10000.0, RopePairing.AdjacentPairs)

    let result = rope backend [| 1; 4 |] [| 1.0f; 0.0f; 0.0f; 1.0f |] [| 1 |] embedding

    let expected = [| cos 1.0; sin 1.0; -(sin 0.01); cos 0.01 |]
    Array.iter2 (fun (e: float) (a: single) -> Assert.Equal(e, float a, 6)) expected result

[<Theory>]
[<InlineData(false)>]
[<InlineData(true)>]
let ``Rope leaves position 0 unchanged`` splitHalves =
    use backend = cpu ()
    let pairing = if splitHalves then RopePairing.SplitHalves else RopePairing.AdjacentPairs
    let values = randomValues (Random(1)) 16
    let embedding = RotaryEmbedding.Standard(8, 10000.0, pairing)

    Assert.Equal<single>(values, rope backend [| 1; 16 |] values [| 0 |] embedding)

[<Theory>]
[<InlineData(false)>]
[<InlineData(true)>]
let ``Rope matches a double-precision reference for several tokens and heads`` splitHalves =
    use backend = cpu ()
    let pairing = if splitHalves then RopePairing.SplitHalves else RopePairing.AdjacentPairs
    let random = Random(2223)
    let positions = [| 0; 1; 7; 1000; 16383; 262143 |]

    // Three heads of eight values, as [tokens; heads × head] and as [tokens; heads; head].
    let smallValues = randomValues random (positions.Length * 24)
    let smallEmbedding = RotaryEmbedding.Standard(8, 10000.0, pairing)
    let smallExpected = ropeReference smallValues positions 8 10000.0 pairing
    assertMatchesReference (rope backend [| positions.Length; 24 |] smallValues positions smallEmbedding) smallExpected
    assertMatchesReference (rope backend [| positions.Length; 3; 8 |] smallValues positions smallEmbedding) smallExpected

    // Ministral queries: 32 heads of 128 values with a frequency base of 1,000,000.
    let queries = randomValues random (2 * 4096)
    let ministral = RotaryEmbedding.Standard(128, 1000000.0, pairing)
    let queryPositions = [| 3; 20000 |]
    let expected = ropeReference queries queryPositions 128 1000000.0 pairing
    assertMatchesReference (rope backend [| 2; 4096 |] queries queryPositions ministral) expected

[<Fact>]
let ``Rope makes query-key products depend only on relative positions`` () =
    use backend = cpu ()
    let random = Random(2425)
    let embedding = RotaryEmbedding.Standard(128, 1000000.0, RopePairing.AdjacentPairs)
    let query = randomValues random 128
    let key = randomValues random 128
    let product (left: single array) (right: single array) = Array.fold2 (fun sum (a: single) (b: single) -> sum + float a * float b) 0.0 left right
    let norm values = sqrt (product values values)

    let scoreAt queryPosition keyPosition =
        let rotatedQuery = rope backend [| 1; 128 |] query [| queryPosition |] embedding
        let rotatedKey = rope backend [| 1; 128 |] key [| keyPosition |] embedding
        Assert.Equal(norm query, norm rotatedQuery, 5)
        product rotatedQuery rotatedKey

    let reference = scoreAt 12 7
    for queryPosition, keyPosition in [ 105, 100; 5005, 5000; 16389, 16384 ] do
        Assert.Equal(reference, scoreAt queryPosition keyPosition, 4)

[<Fact>]
let ``SplitHalves rotates the same pairs as AdjacentPairs after interleaving the halves`` () =
    use backend = cpu ()
    let random = Random(2627)
    let values = randomValues random 8
    // Pair j is (j, j + 4) in split halves and (2j, 2j + 1) once interleaved.
    let interleave (head: single array) = Array.init 8 (fun index -> if index % 2 = 0 then head[index / 2] else head[4 + index / 2])
    let adjacent = RotaryEmbedding.Standard(8, 500.0, RopePairing.AdjacentPairs)
    let split = RotaryEmbedding.Standard(8, 500.0, RopePairing.SplitHalves)

    let splitResult = rope backend [| 1; 8 |] values [| 9 |] split
    let adjacentResult = rope backend [| 1; 8 |] (interleave values) [| 9 |] adjacent

    Assert.Equal<single>(interleave splitResult, adjacentResult)

[<Fact>]
let ``Rope applies the magnitude and supports in-place rotation`` () =
    use backend = cpu ()
    let values = randomValues (Random(2829)) 8
    let doubled = RotaryEmbedding.Create([| 1.0; 0.25; 0.0625; 0.015625 |], 2.0f, RopePairing.AdjacentPairs)

    Assert.Equal<single>(Array.map (fun value -> 2.0f * value) values, rope backend [| 1; 8 |] values [| 0 |] doubled)

    let separate = rope backend [| 2; 8 |] (Array.append values values) [| 3; 11 |] doubled
    use tensor = backend.Upload(TensorShape.Create [| 2; 8 |], ReadOnlyMemory<single>(Array.append values values))
    backend.Rope(tensor, ReadOnlyMemory<int>([| 3; 11 |]), doubled, tensor)
    Assert.Equal<single>(separate, download backend tensor)

[<Fact>]
let ``Rope accepts an empty batch`` () =
    use backend = cpu ()
    let embedding = RotaryEmbedding.Standard(128, 1000000.0, RopePairing.AdjacentPairs)

    Assert.Empty(rope backend [| 0; 128 |] [||] [||] embedding)

[<Fact>]
let ``Rope validates shapes positions ownership and disposal before writing`` () =
    use backend = cpu ()
    use otherBackend = cpu ()
    let embedding = RotaryEmbedding.Standard(4, 10000.0, RopePairing.AdjacentPairs)
    let shape = TensorShape.Create [| 2; 8 |]
    use input = backend.Upload(shape, ReadOnlyMemory<single>(Array.create 16 1.0f))
    use destination = backend.Upload(shape, ReadOnlyMemory<single>(Array.create 16 42.0f))
    use vector = backend.Allocate(TensorShape.Create [| 8 |])
    use partialHeads = backend.Allocate(TensorShape.Create [| 2; 6 |])
    use otherShape = backend.Allocate(TensorShape.Create [| 2; 4 |])
    use foreign = otherBackend.Allocate(shape)
    let disposed = backend.Allocate(shape)
    disposed.Dispose()
    let positions = ReadOnlyMemory<int>([| 0; 1 |])

    Assert.Throws<ArgumentException>(fun () -> backend.Rope(input, ReadOnlyMemory<int>([| 0 |]), embedding, destination)) |> ignore
    Assert.Throws<ArgumentException>(fun () -> backend.Rope(vector, ReadOnlyMemory<int>(Array.zeroCreate 8), embedding, vector)) |> ignore
    let headError = Assert.Throws<ArgumentException>(fun () -> backend.Rope(partialHeads, positions, embedding, partialHeads))
    Assert.Contains("4 values", headError.Message)
    Assert.Throws<ArgumentException>(fun () -> backend.Rope(input, positions, embedding, otherShape)) |> ignore
    let positionError = Assert.Throws<ArgumentOutOfRangeException>(fun () -> backend.Rope(input, ReadOnlyMemory<int>([| 0; -1 |]), embedding, destination))
    Assert.Contains("index 1", positionError.Message)
    Assert.Throws<ArgumentException>(fun () -> backend.Rope(foreign, positions, embedding, destination)) |> ignore
    Assert.Throws<ObjectDisposedException>(fun () -> backend.Rope(disposed, positions, embedding, destination)) |> ignore
    Assert.Throws<ArgumentNullException>(fun () -> backend.Rope(input, positions, Unchecked.defaultof<RotaryEmbedding>, destination)) |> ignore
    Assert.Throws<ArgumentNullException>(fun () -> backend.Rope(Unchecked.defaultof<ITensor>, positions, embedding, destination)) |> ignore
    Assert.Equal<single>(Array.create 16 42.0f, download backend destination)
