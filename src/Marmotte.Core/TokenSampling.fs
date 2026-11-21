namespace Marmotte.Core

open System

/// Failures when selecting a token from vocabulary-ordered logits.
[<RequireQualifiedAccess>]
type TokenSamplingError =
    | EmptyLogits
    /// NaN or positive infinity, with the zero-based vocabulary token ID and offending value.
    | InvalidLogit of tokenId: int * value: single
    /// Every logit is negative infinity, so every token is excluded.
    | NoEligibleTokens

/// Host-side token selection, independent of model execution, tokenization and stopping policy.
[<RequireQualifiedAccess>]
module TokenSampling =
    /// Returns the zero-based index of the largest finite logit. Equal maxima select the smallest
    /// token ID, including ties between signed zeros. Negative infinity excludes a token; an
    /// entirely excluded vocabulary returns NoEligibleTokens. Empty input returns EmptyLogits.
    /// NaN or positive infinity anywhere returns InvalidLogit for the first offending token,
    /// even if an earlier token already has the largest representable finite value.
    /// Reads without modifying or retaining the input. Takes O(vocabulary size) time and constant
    /// auxiliary space, without sorting, softmax, random state or a seed. EOS has no special role.
    let greedy (logits: ReadOnlySpan<single>) : Result<int, TokenSamplingError> =
        if logits.IsEmpty then
            Error TokenSamplingError.EmptyLogits
        else
            let mutable bestToken = -1
            let mutable bestLogit = Single.NegativeInfinity
            let mutable invalidToken = -1
            let mutable token = 0
            while token < logits.Length && invalidToken < 0 do
                let value = logits[token]
                if Single.IsNaN value || Single.IsPositiveInfinity value then
                    invalidToken <- token
                elif value > bestLogit then
                    bestLogit <- value
                    bestToken <- token
                token <- token + 1

            if invalidToken >= 0 then
                Error (TokenSamplingError.InvalidLogit(invalidToken, logits[invalidToken]))
            elif bestToken < 0 then
                Error TokenSamplingError.NoEligibleTokens
            else
                Ok bestToken

/// Immutable controls for probabilistic selection. Use TokenSampling.greedy for deterministic
/// maximum selection rather than setting temperature to zero.
[<Sealed>]
type TokenSamplingParameters private (temperature: double, topK: int option, topP: double) =
    member _.Temperature = temperature
    member _.TopK = topK
    member _.TopP = topP

    /// Temperature must be finite and positive. None disables top-k; Some k requires k > 0
    /// and keeps at most k eligible tokens. Top-p must be finite in (0, 1]; 1 disables it.
    /// Invalid parameters raise ArgumentOutOfRangeException before a sampler is created.
    static member Create(temperature: double, topK: int option, topP: double) =
        if not (Double.IsFinite temperature) || temperature <= 0.0 then
            raise (ArgumentOutOfRangeException(nameof temperature, "Temperature must be finite and positive."))
        match topK with
        | Some count when count <= 0 ->
            raise (ArgumentOutOfRangeException(nameof topK, "Top-k must be positive or None."))
        | _ -> ()
        if not (Double.IsFinite topP) || topP <= 0.0 || topP > 1.0 then
            raise (ArgumentOutOfRangeException(nameof topP, "Top-p must be finite and in (0, 1]."))
        TokenSamplingParameters(temperature, topK, topP)

module private SamplingDistribution =
    /// Rank before temperature scaling: positive temperature preserves logit order. Subtract
    /// the maximum before division so even tiny temperatures cannot overflow a positive score.
    let prepare (parameters: TokenSamplingParameters) (logits: ReadOnlySpan<single>) =
        let candidates = ResizeArray<int * single>()
        for token = 0 to logits.Length - 1 do
            if Single.IsFinite logits[token] then
                candidates.Add(token, logits[token])
        let ranked = candidates.ToArray()
        Array.sortInPlaceWith (fun (leftToken, left) (rightToken, right) ->
            let order = compare right left
            if order = 0 then compare leftToken rightToken else order) ranked
        let count =
            match parameters.TopK with
            | Some count -> min count ranked.Length
            | None -> ranked.Length
        let maximum = float (snd ranked[0])
        let weights = Array.init count (fun index ->
            Math.Exp((float (snd ranked[index]) - maximum) / parameters.Temperature))
        let total = Array.sum weights
        let mutable retained = count
        let mutable retainedWeight = total
        if parameters.TopP < 1.0 then
            let threshold = parameters.TopP * total
            retained <- 0
            retainedWeight <- 0.0
            while retained < count && (retained = 0 || retainedWeight < threshold) do
                retainedWeight <- retainedWeight + weights[retained]
                retained <- retained + 1
        ranked, weights, retained, retainedWeight

/// Stateful categorical token sampler. Each instance owns a System.Random initialized from the
/// explicit seed; create a new instance with the same parameters/seed to restart the sequence.
/// Equal logits and identical successful call sequences are reproducible on the same runtime
/// and platform. No cross-runtime or cross-platform bit-for-bit guarantee. Not thread-safe.
[<Sealed>]
type TokenSampler(parameters: TokenSamplingParameters, seed: int) =
    do ArgumentNullException.ThrowIfNull(parameters)
    let random = Random(seed)

    member _.Parameters = parameters

    /// Validates the entire vocabulary using greedy's error rules, then ranks by descending
    /// logit (smallest token ID on ties), applies top-k, temperature-scaled stable softmax, and
    /// top-p on that filtered distribution. Top-p keeps the smallest non-empty prefix whose
    /// cumulative probability reaches p, including the crossing token. Remaining weights are
    /// renormalized for the draw. Underflowed zero weights are never selected.
    /// Every successful call consumes one random draw, even for a single candidate. Invalid
    /// logits return Error without advancing random state. Input is neither changed nor retained;
    /// token IDs are relative to the supplied vocabulary span. O(V log V) time and O(V) scratch.
    /// Does not handle EOS, stopping, penalties or tokenization. Allocation failures propagate.
    member _.Sample(logits: ReadOnlySpan<single>) : Result<int, TokenSamplingError> =
        match TokenSampling.greedy logits with
        | Error error -> Error error
        | Ok _ ->
            let ranked, weights, retained, total = SamplingDistribution.prepare parameters logits
            let target = random.NextDouble() * total
            let mutable cumulative = 0.0
            let mutable selected = -1
            let mutable lastPositive = 0
            let mutable index = 0
            while index < retained && selected < 0 do
                if weights[index] > 0.0 then
                    lastPositive <- index
                    cumulative <- cumulative + weights[index]
                    if target < cumulative then selected <- index
                index <- index + 1
            // Rounding at the upper boundary must not select a zero-probability tail entry.
            let selected = if selected >= 0 then selected else lastPositive
            Ok (fst ranked[selected])
