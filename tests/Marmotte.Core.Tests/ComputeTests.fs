module Marmotte.Core.ComputeTests

open System
open System.Collections.Generic
open Marmotte.Core
open Xunit

[<Fact>]
let ``execution shapes distinguish scalars empty tensors and matrices`` () =
    let scalar = TensorShape.Create [||]
    let empty = TensorShape.Create [| Int32.MaxValue; Int32.MaxValue; 0 |]
    let matrix = TensorShape.Create [| 2; 3 |]

    Assert.Equal(0, scalar.Rank)
    Assert.Equal(1, scalar.ElementCount)
    Assert.Equal(0, empty.ElementCount)
    Assert.Equal(6, matrix.ElementCount)
    Assert.Equal(TensorShape.Create [| 2; 3 |], matrix)
    Assert.NotEqual(TensorShape.Create [| 3; 2 |], matrix)
    Assert.NotEqual(TensorShape.Create [| 6 |], matrix)

[<Fact>]
let ``execution shapes compare hash and print by dimensions`` () =
    let matrix = TensorShape.Create [| 2; 3 |]
    let same = TensorShape.Create [| 2; 3 |]

    Assert.True((matrix = same))
    Assert.False((matrix = TensorShape.Create [| 3; 2 |]))
    Assert.Equal(matrix.GetHashCode(), same.GetHashCode())
    Assert.False(matrix.Equals(null: obj))
    Assert.False(matrix.Equals(box [| 2; 3 |]))
    Assert.Contains(same, HashSet<TensorShape>([ matrix ]))
    Assert.Equal("[2; 3]", string matrix)
    Assert.Equal("[]", string (TensorShape.Create [||]))

[<Fact>]
let ``execution shape owns immutable dimensions`` () =
    let dimensions = [| 2; 3 |]
    let shape = TensorShape.Create dimensions
    dimensions[0] <- 5

    Assert.Equal<int>([| 2; 3 |], shape.Dimensions)
    let exposed = shape.Dimensions :?> IList<int>
    Assert.Throws<NotSupportedException>(fun () -> exposed[0] <- 7) |> ignore
    Assert.Equal(6, shape.ElementCount)

[<Fact>]
let ``execution shapes reject negative dimensions and overflow without allocating values`` () =
    for dimensions in [ [| -1 |]; [| 0; -1 |]; [| Int32.MaxValue |]; [| 65536; 65536 |] ] do
        Assert.Throws<ArgumentException>(fun () -> TensorShape.Create dimensions |> ignore) |> ignore

    Assert.Throws<ArgumentNullException>(fun () -> TensorShape.Create null |> ignore) |> ignore

[<Fact>]
let ``standard rotary frequencies decrease geometrically from 1`` () =
    let embedding = RotaryEmbedding.Standard(8, 10000.0, RopePairing.AdjacentPairs)

    Assert.Equal(8, embedding.HeadDimension)
    Assert.Equal(1.0f, embedding.Magnitude)
    Assert.Equal(RopePairing.AdjacentPairs, embedding.Pairing)
    // Pair j of 4 rotates at 10000^(-j / 4): 1, 0.1, 0.01 and 0.001 radians per position.
    for pair, expected in [ 0, 1.0; 1, 0.1; 2, 0.01; 3, 0.001 ] do
        Assert.Equal(expected, embedding.Frequencies[pair], 15)

[<Fact>]
let ``custom rotary frequencies are copied and validated`` () =
    let frequencies = [| 1.0; 0.5 |]
    let embedding = RotaryEmbedding.Create(frequencies, 1.25f, RopePairing.SplitHalves)
    frequencies[0] <- 9.0

    Assert.Equal<double>([| 1.0; 0.5 |], embedding.Frequencies)
    Assert.Equal(4, embedding.HeadDimension)
    Assert.Equal(1.25f, embedding.Magnitude)
    Assert.Throws<NotSupportedException>(fun () -> (embedding.Frequencies :?> IList<double>)[0] <- 2.0) |> ignore

    for invalid in [ [||]; [| Double.NaN |]; [| -1.0 |]; [| Double.PositiveInfinity |] ] do
        Assert.Throws<ArgumentException>(fun () -> RotaryEmbedding.Create(invalid, 1.0f, RopePairing.AdjacentPairs) |> ignore) |> ignore
    for magnitude in [ 0.0f; -1.0f; Single.NaN; Single.PositiveInfinity ] do
        Assert.Throws<ArgumentOutOfRangeException>(fun () -> RotaryEmbedding.Create([| 1.0 |], magnitude, RopePairing.AdjacentPairs) |> ignore) |> ignore

[<Fact>]
let ``standard rotary embedding rejects odd heads and invalid bases`` () =
    for headDimension in [ 0; -2; 3 ] do
        Assert.Throws<ArgumentException>(fun () -> RotaryEmbedding.Standard(headDimension, 10000.0, RopePairing.AdjacentPairs) |> ignore) |> ignore
    for frequencyBase in [ 0.0; -10.0; Double.NaN; Double.PositiveInfinity ] do
        Assert.Throws<ArgumentOutOfRangeException>(fun () -> RotaryEmbedding.Standard(4, frequencyBase, RopePairing.AdjacentPairs) |> ignore) |> ignore

[<Fact>]
let ``attention temperature is 1 within the first floor and grows logarithmically after it`` () =
    let temperature = AttentionTemperature.Create(float 0.1f, 16384)

    Assert.Equal(1.0f, temperature.Factor 0)
    Assert.Equal(1.0f, temperature.Factor 16383)
    // 1 + 0.1 ln 2, 1 + 0.1 ln 3 and 1 + 0.1 ln 16, computed by hand.
    Assert.Equal(1.06931, float (temperature.Factor 16384), 5)
    Assert.Equal(temperature.Factor 16384, temperature.Factor 32767)
    Assert.Equal(1.10986, float (temperature.Factor 32768), 5)
    Assert.Equal(1.27726, float (temperature.Factor 262143), 5)
    Assert.Equal(16384, temperature.FloorLength)

[<Fact>]
let ``attention temperature validates its parameters and positions`` () =
    let disabled = AttentionTemperature.Create(0.0, 16384)
    Assert.Equal(1.0f, disabled.Factor Int32.MaxValue)

    let temperature = AttentionTemperature.Create(0.1, 10)
    Assert.Throws<ArgumentOutOfRangeException>(fun () -> temperature.Factor -1 |> ignore) |> ignore
    for scale in [ -0.1; Double.NaN; Double.PositiveInfinity ] do
        Assert.Throws<ArgumentOutOfRangeException>(fun () -> AttentionTemperature.Create(scale, 10) |> ignore) |> ignore
    for floorLength in [ 0; -1 ] do
        Assert.Throws<ArgumentOutOfRangeException>(fun () -> AttentionTemperature.Create(0.1, floorLength) |> ignore) |> ignore
