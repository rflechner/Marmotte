namespace Marmotte.Core

open System
open System.Collections.Generic
open System.Threading

/// Why a generation loop ended. Sampling failure is distinct from successful stopping.
[<RequireQualifiedAccess>]
type GenerationStopReason =
    | EndToken of tokenId: int
    | TokenLimit
    | ContextLimit
    | SamplingFailed of TokenSamplingError

/// Caller-owned generated IDs, excluding prompt and end tokens, and the reason the loop stopped.
/// Tokens already delivered to onToken are included here, including on a sampling failure.
type GenerationResult =
    { Tokens: int array
      StopReason: GenerationStopReason }

/// Orchestrates one prompt and continuation through a borrowed session and selection policy.
/// Owns no disposable resources. The session must be new or reset before Generate; it is never
/// reset or disposed automatically. The selector may own random state; resetting a session does
/// not reset that state. Calls and callbacks must not reenter or concurrently use the session.
[<Sealed>]
type TokenGenerator(session: GenerationSession, maximumNewTokens: int,
                    endTokenIds: ReadOnlyMemory<int>,
                    selectToken: ReadOnlyMemory<single> -> Result<int, TokenSamplingError>) =
    do
        ArgumentNullException.ThrowIfNull(session)
        ArgumentNullException.ThrowIfNull(selectToken)
        if maximumNewTokens < 0 then
            raise (ArgumentOutOfRangeException(nameof maximumNewTokens, "The generated-token limit must be non-negative."))

    let endTokens = HashSet<int>()
    do
        for token in endTokenIds.Span do
            if token < 0 || token >= session.VocabularySize then
                raise (ArgumentOutOfRangeException(nameof endTokenIds, "End token IDs must belong to the vocabulary."))
            endTokens.Add token |> ignore

    /// Prefills the non-empty prompt, then selects at most maximumNewTokens ordinary tokens.
    /// End IDs are copied at construction (duplicates are harmless; empty disables EOS stopping).
    /// A selected end token is neither emitted, returned nor decoded; EOS in the prompt does not
    /// stop generation. EndToken takes precedence over limits when it is selected.
    /// A zero limit still validates and prefills the prompt, but never calls the selector.
    ///
    /// onToken runs synchronously once per ordinary token, before decoding it. The final token
    /// is not decoded when the token limit is reached or cache capacity is exhausted. A full
    /// prompt cache can therefore produce one token from its last logits. TokenLimit takes
    /// precedence over ContextLimit when both apply. No extra selection or forward call occurs
    /// after stopping. Tokens counts exclude prompt and EOS; the cache excludes the last emitted
    /// token when it was not decoded. The selector must consume logits before returning.
    ///
    /// Sampling Error returns SamplingFailed with the completed output; an out-of-vocabulary
    /// Ok ID raises ArgumentOutOfRangeException before emission. Callback/selector exceptions and
    /// cancellation propagate without returning a result; previously delivered tokens remain
    /// delivered. Reset the session before another Generate call. Only session execution errors
    /// fault its cache. Cancellation is checked before prefill, selection and decoding, not inside
    /// kernels or callbacks. Session lifetime, validation and execution errors propagate.
    member _.Generate(prompt: ReadOnlyMemory<int>, ?onToken: (int -> unit),
                      ?cancellationToken: CancellationToken) : GenerationResult =
        let cancellation = defaultArg cancellationToken CancellationToken.None
        let emit = defaultArg onToken ignore
        ArgumentNullException.ThrowIfNull(emit)
        cancellation.ThrowIfCancellationRequested()
        let mutable logits = session.Prefill(prompt, cancellation)
        let output = ResizeArray<int>()
        let mutable reason = if maximumNewTokens = 0 then Some GenerationStopReason.TokenLimit else None
        while reason.IsNone do
            cancellation.ThrowIfCancellationRequested()
            match selectToken (ReadOnlyMemory<single>(logits)) with
            | Error error -> reason <- Some (GenerationStopReason.SamplingFailed error)
            | Ok token ->
                if token < 0 || token >= session.VocabularySize then
                    raise (ArgumentOutOfRangeException("selectedToken", "The selected token must belong to the vocabulary."))
                if endTokens.Contains token then
                    reason <- Some (GenerationStopReason.EndToken token)
                else
                    output.Add token
                    emit token
                    if output.Count = maximumNewTokens then
                        reason <- Some GenerationStopReason.TokenLimit
                    elif session.ProcessedTokenCount = session.Capacity then
                        reason <- Some GenerationStopReason.ContextLimit
                    else
                        cancellation.ThrowIfCancellationRequested()
                        logits <- session.Decode(token, cancellation)
        { Tokens = output.ToArray()
          StopReason = reason.Value }
