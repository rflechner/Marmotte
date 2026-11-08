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
