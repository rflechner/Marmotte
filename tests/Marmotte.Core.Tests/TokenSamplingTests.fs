module Marmotte.Core.TokenSamplingTests

open System
open Marmotte.Core
open Xunit

let private greedy (values: single array) = TokenSampling.greedy (ReadOnlySpan<single>(values))

[<Fact>]
let ``greedy selects the largest logit without requiring probabilities`` () =
    Assert.Equal(Ok 2, greedy [| -9.0f; -3.0f; -1.0f; -5.0f |])
    Assert.Equal(Ok 0, greedy [| Single.MaxValue; 100.0f; -Single.MaxValue |])
    Assert.Equal(Ok 2, greedy [| 0.0f; -1.0f; Single.Epsilon |])
    Assert.Equal(Ok 0, greedy [| -100.0f |])

[<Fact>]
let ``greedy resolves repeated maxima and signed zeros by smallest token ID`` () =
    Assert.Equal(Ok 1, greedy [| 1.0f; 8.0f; 8.0f; 2.0f; 8.0f |])
    Assert.Equal(Ok 0, greedy [| -0.0f; 0.0f |])
    Assert.Equal(Ok 0, greedy [| 0.0f; -0.0f |])
    Assert.Equal(Ok 0, greedy [| -2.0f; -2.0f; -2.0f |])

[<Fact>]
let ``negative infinity excludes tokens including the first and last entries`` () =
    Assert.Equal(Ok 1, greedy [| Single.NegativeInfinity; -Single.MaxValue; Single.NegativeInfinity |])
    Assert.Equal(Error TokenSamplingError.NoEligibleTokens, greedy [| Single.NegativeInfinity; Single.NegativeInfinity |])
    Assert.Equal(Error TokenSamplingError.NoEligibleTokens, greedy [| Single.NegativeInfinity |])

[<Fact>]
let ``empty logits report an explicit error`` () =
    Assert.Equal(Error TokenSamplingError.EmptyLogits, greedy Array.empty)

[<Fact>]
let ``nonfinite logits are rejected even after a maximal finite candidate`` () =
    for invalid in [ Single.NaN; Single.PositiveInfinity ] do
        for index in [ 0; 1; 2 ] do
            let values = [| Single.MaxValue; 1.0f; 0.0f |]
            values[index] <- invalid
            match greedy values with
            | Error (TokenSamplingError.InvalidLogit(tokenId, value)) ->
                Assert.Equal(index, tokenId)
                Assert.Equal(BitConverter.SingleToInt32Bits invalid, BitConverter.SingleToInt32Bits value)
            | result -> failwithf "Expected an invalid logit, got %A" result

[<Fact>]
let ``the first invalid entry is reported`` () =
    match greedy [| 1.0f; Single.PositiveInfinity; Single.NaN |] with
    | Error (TokenSamplingError.InvalidLogit(1, value)) -> Assert.True(Single.IsPositiveInfinity value)
    | result -> failwithf "Expected the first invalid logit, got %A" result

[<Fact>]
let ``selection reads only the supplied span and preserves the input`` () =
    let values = [| Single.NaN; -3.0f; 2.0f; 1.0f; Single.PositiveInfinity |]
    let original = values |> Array.map BitConverter.SingleToInt32Bits
    Assert.Equal(Ok 1, TokenSampling.greedy (ReadOnlySpan<single>(values, 1, 3)))
    Assert.Equal<int>(original, values |> Array.map BitConverter.SingleToInt32Bits)

[<Fact>]
let ``greedy agrees with a sorted reference for deterministic finite vocabularies`` () =
    let random = Random(1303)
    for length in [ 1; 2; 31; 256; 131072 ] do
        let values = Array.init length (fun _ -> single (random.Next(-100, 101)))
        let expected =
            values
            |> Array.mapi (fun token value -> token, value)
            |> Array.sortBy (fun (token, value) -> -value, token)
            |> Array.head
            |> fst
        Assert.Equal(Ok expected, greedy values)
        Assert.Equal(Ok expected, greedy values)

let private sampler temperature topK topP seed =
    TokenSampler(TokenSamplingParameters.Create(temperature, topK, topP), seed)

let private sample (sampler: TokenSampler) (values: single array) =
    match sampler.Sample(ReadOnlySpan<single>(values)) with
    | Ok token -> token
    | Error error -> failwithf "Unexpected sampling error: %A" error

[<Fact>]
let ``sampling parameters reject invalid values before use`` () =
    for temperature in [ 0.0; -1.0; Double.NaN; Double.PositiveInfinity; Double.NegativeInfinity ] do
        Assert.Throws<ArgumentOutOfRangeException>(fun () -> sampler temperature None 1.0 0 |> ignore) |> ignore
    for topK in [ 0; -1 ] do
        Assert.Throws<ArgumentOutOfRangeException>(fun () -> sampler 1.0 (Some topK) 1.0 0 |> ignore) |> ignore
    for topP in [ 0.0; -0.1; 1.01; Double.NaN; Double.PositiveInfinity; Double.NegativeInfinity ] do
        Assert.Throws<ArgumentOutOfRangeException>(fun () -> sampler 1.0 None topP 0 |> ignore) |> ignore
    Assert.Throws<ArgumentNullException>(fun () -> TokenSampler(Unchecked.defaultof<_>, 0) |> ignore) |> ignore

[<Fact>]
let ``uniform sampling follows one seeded draw per successful call`` () =
    let seed = 13031
    let sampling = sampler 1.0 None 1.0 seed
    let expectedRandom = Random(seed)
    for _ = 1 to 200 do
        let expected = int (expectedRandom.NextDouble() * 4.0)
        Assert.Equal(expected, sample sampling [| 0.0f; 0.0f; 0.0f; 0.0f |])
    Assert.Equal(0, sample sampling [| 3.0f |])
    expectedRandom.NextDouble() |> ignore
    Assert.Equal(int (expectedRandom.NextDouble() * 4.0), sample sampling [| 0.0f; 0.0f; 0.0f; 0.0f |])

[<Fact>]
let ``equal seeds reproduce a sequence and separate samplers do not share state`` () =
    let draw seed =
        let sampling = sampler 0.8 (Some 3) 0.9 seed
        Array.init 100 (fun _ -> sample sampling [| -1.0f; 0.0f; 0.5f; 0.2f |])
    let expected = draw 42
    draw 17 |> ignore
    Assert.Equal<int>(expected, draw 42)
    Assert.False((expected = draw 43))

[<Fact>]
let ``temperature scales the categorical distribution before nucleus filtering`` () =
    // exp(log(4)/2) = 2: probabilities are 1/3 and 2/3 at temperature 2.
    let values = [| 0.0f; single (log 4.0) |]
    let sampling = sampler 2.0 None 1.0 122
    let counts = Array.zeroCreate<int> 2
    for _ = 1 to 20000 do
        let token = sample sampling values
        counts[token] <- counts[token] + 1
    Assert.InRange(float counts[1] / 20000.0, 0.65, 0.685)
    // At T=1 the leading probability is .8 and alone exceeds .75;
    // at T=2 it is 2/3, so the lower token must remain eligible.
    let cold = sampler 1.0 None 0.75 122
    let warm = sampler 2.0 None 0.75 122
    Assert.All(Array.init 100 (fun _ -> sample cold values), fun token -> Assert.Equal(1, token))
    Assert.Contains(0, Array.init 100 (fun _ -> sample warm values))

[<Fact>]
let ``top-k ranks ties by token ID and top-p uses the top-k distribution`` () =
    let sampling = sampler 1.0 (Some 3) 0.6 77
    // Top-k retains 0,1,2; top-p reaches .6 with 0,1 (not three of the original four).
    let results = Array.init 200 (fun _ -> sample sampling [| 0.0f; 0.0f; 0.0f; 0.0f |])
    Assert.All(results, fun token -> Assert.InRange(token, 0, 1))
    Assert.Contains(0, results)
    Assert.Contains(1, results)
    let one = sampler 1.0 (Some 1) 1.0 77
    Assert.Equal(1, sample one [| -3.0f; 4.0f; 4.0f |])

[<Fact>]
let ``nucleus keeps the crossing token and handles exact boundaries`` () =
    let values = [| 0.0f; 0.0f; 0.0f; 0.0f |]
    let atBoundary = sampler 1.0 None 0.5 19
    let pastBoundary = sampler 1.0 None 0.51 19
    let tiny = sampler 1.0 None Double.Epsilon 19
    Assert.All(Array.init 200 (fun _ -> sample atBoundary values), fun token -> Assert.InRange(token, 0, 1))
    let results = Array.init 200 (fun _ -> sample pastBoundary values)
    Assert.All(results, fun token -> Assert.InRange(token, 0, 2))
    Assert.Contains(2, results)
    Assert.Equal(0, sample tiny values)

[<Fact>]
let ``invalid logits preserve random state even when filters would remove them`` () =
    let sampling = sampler 1.0 (Some 2) 0.9 69
    let control = sampler 1.0 (Some 2) 0.9 69
    Assert.Equal(Error TokenSamplingError.EmptyLogits, sampling.Sample(ReadOnlySpan<single>.Empty))
    Assert.Equal(Error TokenSamplingError.NoEligibleTokens, sampling.Sample(ReadOnlySpan<single>([| Single.NegativeInfinity |])))
    for invalid in [ Single.NaN; Single.PositiveInfinity ] do
        match sampling.Sample(ReadOnlySpan<single>([| Single.MaxValue; 0.0f; invalid |])) with
        | Error (TokenSamplingError.InvalidLogit(2, value)) ->
            Assert.Equal(BitConverter.SingleToInt32Bits invalid, BitConverter.SingleToInt32Bits value)
        | result -> failwithf "Expected invalid logit, got %A" result
    for _ = 1 to 100 do
        Assert.Equal(sample control [| 0.0f; 0.0f |], sample sampling [| 0.0f; 0.0f |])

[<Fact>]
let ``sampling remains finite for extreme logits and temperatures and never picks excluded tokens`` () =
    for temperature in [ Double.Epsilon; 1.0; Double.MaxValue ] do
        let sampling = sampler temperature (Some Int32.MaxValue) 1.0 55
        for _ = 1 to 100 do
            let token = sample sampling [| Single.NegativeInfinity; -Single.MaxValue; Single.MaxValue |]
            Assert.InRange(token, 1, 2)
            if temperature <> Double.MaxValue then Assert.Equal(2, token)
    let tiny = sampler Double.Epsilon None 1.0 55
    Assert.Equal(1, sample tiny [| -Single.MaxValue; -Single.MaxValue / 2.0f |])

[<Fact>]
let ``sampling preserves logits and uses IDs relative to the supplied span`` () =
    let sampling = sampler 1.0 (Some 1) 1.0 55
    let values = [| Single.NaN; -3.0f; 4.0f; 2.0f; Single.PositiveInfinity |]
    let original = Array.map BitConverter.SingleToInt32Bits values
    Assert.Equal(Ok 1, sampling.Sample(ReadOnlySpan<single>(values, 1, 3)))
    Assert.Equal<int>(original, Array.map BitConverter.SingleToInt32Bits values)
