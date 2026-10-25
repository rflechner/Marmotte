module Marmotte.Core.YarnTests

open System
open Marmotte.Core
open Xunit

let private target =
    { ScalingFactor = 16.0
      OriginalContextLength = 16384
      BetaFast = 32.0
      BetaSlow = 1.0
      LogMultiplier = 1.0
      AttentionFactor = 1.0 }

let private prepare parameters =
    RotaryEmbedding.Yarn(128, 1000000.0, RopePairing.AdjacentPairs, parameters)

[<Fact>]
let ``target YaRN preserves fast pairs interpolates slow pairs and blends the middle`` () =
    let embedding = prepare target

    Assert.Equal(128, embedding.HeadDimension)
    Assert.Equal(1.0f, embedding.Magnitude)
    Assert.Equal(RopePairing.AdjacentPairs, embedding.Pairing)
    // Solving L * base^(-2j/D) / (2*pi) = beta gives 20.3849609879 and 36.4398940900.
    // The integer transition runs from pair 20 through pair 37, with 17 equal steps.
    for pair in 0 .. 63 do
        let original = Math.Exp(-float pair * Math.Log(1000000.0) / 64.0)
        let multiplier =
            if pair <= 20 then 1.0
            elif pair >= 37 then 1.0 / 16.0
            else 1.0 - 15.0 * float (pair - 20) / (16.0 * 17.0)
        Assert.InRange(abs (embedding.Frequencies[pair] / original - multiplier), 0.0, 3e-15)

[<Fact>]
let ``YaRN with no extension has standard frequencies and the requested attention factor`` () =
    for pairing in [ RopePairing.AdjacentPairs; RopePairing.SplitHalves ] do
        let standard = RotaryEmbedding.Standard(8, 10000.0, pairing)
        let embedding = RotaryEmbedding.Yarn(8, 10000.0, pairing, { target with ScalingFactor = 1.0; AttentionFactor = 1.25 })
        Assert.Equal(pairing, embedding.Pairing)
        Assert.Equal(1.25f, embedding.Magnitude)
        for pair in 0 .. 3 do
            Assert.Equal(standard.Frequencies[pair], embedding.Frequencies[pair], 15)

[<Theory>]
[<InlineData(0.0, 1.2772588722239781)>]
[<InlineData(0.5, 1.121751143713058)>]
[<InlineData(1.0, 1.0)>]
[<InlineData(2.0, 0.8216431602571131)>]
let ``YaRN combines logarithmic magnitude with the model attention factor`` logMultiplier expected =
    let embedding = prepare { target with LogMultiplier = logMultiplier; AttentionFactor = 1.5 }
    Assert.Equal(single (1.5 * expected), embedding.Magnitude)

[<Fact>]
let ``YaRN handles a collapsed transition without division by zero`` () =
    // With D=4, base=10000, L=1 and both betas=1, both bounded cutoffs are zero.
    let parameters = { target with OriginalContextLength = 1; BetaFast = 1.0; ScalingFactor = 4.0 }
    let embedding = RotaryEmbedding.Yarn(4, 10000.0, RopePairing.AdjacentPairs, parameters)
    Assert.Equal<double>([| 1.0; 0.0025 |], embedding.Frequencies)

[<Fact>]
let ``YaRN bounds the upper cutoff by D - 1 even when the transition extends past the last pair`` () =
    // D=8, base=100, L=1000 gives raw cutoffs 2.4036 and 4.4036, rounded to 2 and 5.
    // Clamping the upper cutoff to D/2-1=3 instead of D-1 would wrongly fully scale pair 3.
    let parameters = { target with OriginalContextLength = 1000; ScalingFactor = 4.0; BetaFast = 10.0 }
    let embedding = RotaryEmbedding.Yarn(8, 100.0, RopePairing.SplitHalves, parameters)
    let originalLast = Math.Pow(100.0, -0.75)
    Assert.Equal(originalLast * 0.75, embedding.Frequencies[3], 15)

    // Both cutoffs beyond the available pairs: every frequency stays unchanged.
    let outside = RotaryEmbedding.Yarn(8, 100.0, RopePairing.SplitHalves, { parameters with OriginalContextLength = Int32.MaxValue })
    Assert.Equal<double>(RotaryEmbedding.Standard(8, 100.0, RopePairing.SplitHalves).Frequencies, outside.Frequencies)

[<Fact>]
let ``YaRN rejects invalid context dimensions bases and null configuration`` () =
    for headDimension in [ 0; -2; 3 ] do
        Assert.Throws<ArgumentException>(fun () -> RotaryEmbedding.Yarn(headDimension, 10000.0, RopePairing.AdjacentPairs, target) |> ignore) |> ignore
    for frequencyBase in [ Double.NaN; Double.PositiveInfinity; Double.NegativeInfinity; -1.0; 0.0; 0.5; 1.0 ] do
        Assert.Throws<ArgumentOutOfRangeException>(fun () -> RotaryEmbedding.Yarn(4, frequencyBase, RopePairing.AdjacentPairs, target) |> ignore) |> ignore
    for contextLength in [ 0; -1 ] do
        Assert.Throws<ArgumentException>(fun () -> prepare { target with OriginalContextLength = contextLength } |> ignore) |> ignore
    Assert.Throws<ArgumentNullException>(fun () -> prepare Unchecked.defaultof<YarnParameters> |> ignore) |> ignore

[<Fact>]
let ``YaRN rejects invalid numeric parameters`` () =
    let setters =
        [ (fun value -> { target with ScalingFactor = value })
          (fun value -> { target with BetaFast = value })
          (fun value -> { target with BetaSlow = value })
          (fun value -> { target with LogMultiplier = value })
          (fun value -> { target with AttentionFactor = value }) ]
    for set in setters do
        for value in [ Double.NaN; Double.PositiveInfinity; Double.NegativeInfinity; -1.0 ] do
            Assert.Throws<ArgumentOutOfRangeException>(fun () -> prepare (set value) |> ignore) |> ignore
    for parameters in
        [ { target with ScalingFactor = 0.5 }
          { target with BetaSlow = 0.0 }
          { target with BetaFast = 0.5 }
          { target with BetaSlow = 33.0 }
          { target with AttentionFactor = 0.0 } ] do
        Assert.Throws<ArgumentOutOfRangeException>(fun () -> prepare parameters |> ignore) |> ignore

[<Fact>]
let ``YaRN rejects magnitudes that overflow or underflow F32`` () =
    for attentionFactor in [ Double.MaxValue; Double.Epsilon ] do
        Assert.Throws<ArgumentException>(fun () -> prepare { target with AttentionFactor = attentionFactor } |> ignore) |> ignore

[<Fact>]
let ``YaRN finite extreme thresholds do not overflow the cutoff calculation`` () =
    for threshold in [ Double.Epsilon; Double.MaxValue ] do
        let embedding = prepare { target with BetaFast = threshold; BetaSlow = threshold }
        Assert.All(embedding.Frequencies, fun value -> Assert.True(Double.IsFinite value && value >= 0.0))
