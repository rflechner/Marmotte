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

/// The target model: 26 layers with 8 key-value heads of 128 values, for a 4,096-token context.
let private ministralCache keyEncoding valueEncoding =
    KeyValueCacheLayout.Create(
        layers = 26,
        capacity = 4096,
        keyValueHeads = 8,
        headDimension = 128,
        keyEncoding = keyEncoding,
        valueEncoding = valueEncoding)

[<Fact>]
let ``key-value cache layouts size the target model's cache for each encoding`` () =
    // Each position stores 26 key rows and 26 value rows of 1,024 values: 208 KiB in F32.
    let f32 = ministralCache TensorEncoding.F32 TensorEncoding.F32
    Assert.Equal(1024, f32.RowValues)
    Assert.Equal(212992L, f32.BytesPerPosition)
    Assert.Equal(832L * 1024L * 1024L, f32.TotalBytes)

    // A row holds 32 blocks of 32 values: 34 bytes each in Q8_0 and 18 in Q4_0.
    let cases =
        [ TensorEncoding.F16, TensorEncoding.F16, 2048, 2048, 106496L
          TensorEncoding.BF16, TensorEncoding.BF16, 2048, 2048, 106496L
          TensorEncoding.Q8_0, TensorEncoding.Q8_0, 1088, 1088, 56576L
          TensorEncoding.Q4_0, TensorEncoding.Q4_0, 576, 576, 29952L
          TensorEncoding.Q8_0, TensorEncoding.Q4_0, 1088, 576, 43264L ]

    for keyEncoding, valueEncoding, keyRowBytes, valueRowBytes, bytesPerPosition in cases do
        let layout = ministralCache keyEncoding valueEncoding
        Assert.Equal(keyRowBytes, layout.KeyRowBytes)
        Assert.Equal(valueRowBytes, layout.ValueRowBytes)
        Assert.Equal(bytesPerPosition, layout.BytesPerPosition)
        Assert.Equal(bytesPerPosition * 4096L, layout.TotalBytes)

[<Fact>]
let ``key-value cache layouts reject invalid dimensions encodings and sizes`` () =
    let create layers capacity heads headDimension keyEncoding valueEncoding =
        KeyValueCacheLayout.Create(layers, capacity, heads, headDimension, keyEncoding, valueEncoding) |> ignore

    for layers, capacity, heads, headDimension in [ 0, 1, 1, 32; 1, 0, 1, 32; 1, 1, -1, 32; 1, 1, 1, 0 ] do
        Assert.Throws<ArgumentOutOfRangeException>(fun () -> create layers capacity heads headDimension TensorEncoding.F32 TensorEncoding.F32) |> ignore

    Assert.Throws<NotSupportedException>(fun () -> create 1 1 1 256 TensorEncoding.Q4_K TensorEncoding.F32) |> ignore
    Assert.Throws<NotSupportedException>(fun () -> create 1 1 1 256 TensorEncoding.F32 TensorEncoding.Q6_K) |> ignore

    // A Q8_0 block of 32 values would span two heads of 16.
    let spanning = Assert.Throws<ArgumentException>(fun () -> create 1 1 8 16 TensorEncoding.F16 TensorEncoding.Q8_0)
    Assert.Equal("valueEncoding", spanning.ParamName)

    // One layer of 524,288 F32 rows of 4,096 bytes needs 2 GiB, beyond a managed array; F16 needs half.
    Assert.Throws<ArgumentException>(fun () -> create 1 524288 8 128 TensorEncoding.F32 TensorEncoding.F16) |> ignore
    create 1 524288 8 128 TensorEncoding.F16 TensorEncoding.F16
    Assert.Throws<ArgumentException>(fun () -> create 1 1 65536 65536 TensorEncoding.F16 TensorEncoding.F16) |> ignore
