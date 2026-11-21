namespace Marmotte.Core

open System
open System.Threading

[<RequireQualifiedAccess>]
type private GenerationState =
    | Empty
    | Ready
    | Faulted

/// One sequence's prompt processing and incremental decoding. Owns its KV cache and borrows
/// the loaded model, backend and source, which must remain alive. Not thread-safe. Sampling,
/// text tokenization/decoding and EOS policy belong to callers. Create one session per sequence.
[<Sealed>]
type GenerationSession private (model: TransformerModel, layout: KeyValueCacheLayout,
                                batchSize: int, initialCache: IKeyValueCache) =
    let backend = model.ExecutionBackend
    let mutable cache = initialCache
    let mutable state = GenerationState.Empty
    let mutable processedTokens = 0
    let mutable disposed = false

    let ensureAlive () =
        ObjectDisposedException.ThrowIf(disposed, typeof<GenerationSession>)
        model.EnsureAlive()

    let requireToken token =
        if token < 0 || token >= model.VocabularySize then
            raise (ArgumentOutOfRangeException("tokenIds", "Every token ID must belong to the vocabulary."))

    // Synchronize every batch, including cache-only prefill, so failures cannot be
    // attributed to a later call after we have reported a successful prefix length.
    let execute (tokens: ReadOnlyMemory<int>) returnLastLogits =
        let selection = if returnLastLogits then LogitSelection.LastToken else LogitSelection.NoTokens
        use output = model.Forward(cache, processedTokens, tokens, selection)
        let result =
            if returnLastLogits then
                let host = Array.zeroCreate<single> output.Shape.ElementCount
                backend.Download(output, host.AsMemory())
                host
            else
                backend.Synchronize()
                Array.empty
        processedTokens <- processedTokens + tokens.Length
        result

    /// Number of tokens processed successfully, including the prompt and Decode inputs. Merely
    /// selecting/emitting a token does not count it: it enters the cache on the next Decode call.
    /// On a fault this counts only completed batches; the partial cache must not be reused.
    member _.ProcessedTokenCount = processedTokens
    member _.VocabularySize = model.VocabularySize
    member _.Capacity = layout.Capacity
    member _.BatchSize = batchSize
    member _.IsFaulted = state = GenerationState.Faulted

    /// Processes a non-empty prompt from position zero in batches of at most BatchSize tokens.
    /// Requires a new or reset session. Returns a caller-owned F32 array of VocabularySize logits
    /// predicting the first continuation token, from the final prompt position. Intermediate
    /// batches request cache-only execution; the final batch requests only its last logits,
    /// so vocabulary-sized outputs are not materialized/downloaded for earlier positions. All prompt IDs
    /// and total capacity are validated before execution. The input must not be mutated during
    /// the call and is no longer retained after return. Empty prompts/invalid state raise
    /// ArgumentException/InvalidOperationException; invalid IDs/capacity raise
    /// ArgumentOutOfRangeException. No BOS or EOS is inserted automatically.
    /// Cancellation is checked before execution and between completed batches, not inside kernels.
    /// Cancellation or any execution failure after processing starts faults the session; call
    /// Reset or dispose it. Pre-execution validation/cancellation leaves it unchanged.
    member _.Prefill(tokenIds: ReadOnlyMemory<int>, ?cancellationToken: CancellationToken) =
        ensureAlive ()
        if state <> GenerationState.Empty then
            invalidOp "Prefill requires a new or reset session."
        if tokenIds.IsEmpty then invalidArg (nameof tokenIds) "A prompt must contain at least one token."
        if tokenIds.Length > layout.Capacity then
            raise (ArgumentOutOfRangeException(nameof tokenIds, "The prompt exceeds session capacity."))
        for token in tokenIds.Span do requireToken token
        let cancellation = defaultArg cancellationToken CancellationToken.None
        cancellation.ThrowIfCancellationRequested()
        try
            let mutable result = Array.empty<single>
            while processedTokens < tokenIds.Length do
                cancellation.ThrowIfCancellationRequested()
                let count = min batchSize (tokenIds.Length - processedTokens)
                let last = count = tokenIds.Length - processedTokens
                result <- execute (tokenIds.Slice(processedTokens, count)) last
            state <- GenerationState.Ready
            result
        with _ ->
            state <- GenerationState.Faulted
            reraise ()

    /// Appends one caller-selected token at ProcessedTokenCount and returns caller-owned logits
    /// predicting the following token. Requires successful Prefill. Validates the ID and remaining
    /// capacity before execution; a full cache raises ArgumentOutOfRangeException, with no rolling
    /// window or eviction. This does not sample, detect EOS or emit text. Cancellation is checked
    /// before execution only; a running kernel completes normally. Execution errors fault the
    /// session; validation errors and pre-execution cancellation leave it usable.
    member _.Decode(tokenId: int, ?cancellationToken: CancellationToken) =
        ensureAlive ()
        if state <> GenerationState.Ready then
            invalidOp "Decode requires a successful Prefill and a session without execution failures."
        requireToken tokenId
        if processedTokens = layout.Capacity then
            raise (ArgumentOutOfRangeException(nameof tokenId, "The session cache is full."))
        let cancellation = defaultArg cancellationToken CancellationToken.None
        cancellation.ThrowIfCancellationRequested()
        try
            execute (ReadOnlyMemory<int>([| tokenId |])) true
        with _ ->
            state <- GenerationState.Faulted
            reraise ()

    /// Discards the sequence and clears a fault by replacing the cache. Allocation happens first;
    /// if it fails, the existing state is retained. This briefly requires two caches' memory.
    /// Does not reload weights or dispose the borrowed model/backend/source.
    member _.Reset() =
        ensureAlive ()
        let replacement = backend.AllocateKeyValueCache(layout)
        try cache.Dispose()
        with _ ->
            replacement.Dispose()
            state <- GenerationState.Faulted
            reraise ()
        cache <- replacement
        processedTokens <- 0
        state <- GenerationState.Empty

    /// Allocates a sequence cache with explicit capacity and K/V encodings. Capacity must be in
    /// [1, model.ContextLength], and batchSize must be positive. KeyValueCacheLayout validates
    /// encoding block compatibility. Invalid limits raise ArgumentOutOfRangeException; disposed
    /// models raise ObjectDisposedException; allocation/backend failures propagate. Model and
    /// source ownership stay with the caller. No vocabulary-sized tensor is allocated here.
    static member Create(model: TransformerModel, capacity: int, batchSize: int,
                         keyEncoding: TensorEncoding, valueEncoding: TensorEncoding) =
        ArgumentNullException.ThrowIfNull(model)
        model.EnsureAlive()
        let configuration = model.Configuration
        if capacity <= 0 || capacity > configuration.ContextLength then
            raise (ArgumentOutOfRangeException(nameof capacity, "Capacity must fit the model context."))
        if batchSize <= 0 then
            raise (ArgumentOutOfRangeException(nameof batchSize, "Prompt batch size must be positive."))
        let layout = KeyValueCacheLayout.Create(configuration.BlockCount, capacity,
                                               configuration.KeyValueHeadCount, configuration.KeyLength,
                                               keyEncoding, valueEncoding)
        let cache = model.ExecutionBackend.AllocateKeyValueCache(layout)
        new GenerationSession(model, layout, batchSize, cache)

    interface IDisposable with
        member _.Dispose() =
            if not disposed then
                disposed <- true
                cache.Dispose()
