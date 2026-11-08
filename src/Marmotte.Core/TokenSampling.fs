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
