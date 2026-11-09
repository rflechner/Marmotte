namespace Marmotte.Core

open System
open System.IO

/// Positions for which a forward pass produces vocabulary logits. Every mode
/// still populates every layer's key/value cache for all supplied tokens.
[<RequireQualifiedAccess>]
type LogitSelection =
    /// Return one vocabulary row for every input token.
    | AllTokens
    /// Return only the last input token's vocabulary row, or zero rows for empty input.
    | LastToken
    /// Populate the cache without computing vocabulary logits.
    | NoTokens

/// Borrowed buffers for one feed-forward execution, all with the same token count.
type private FeedForwardScratch =
    { Normalized: ITensor
      Gate: ITensor
      Up: ITensor
      Residual: ITensor }

/// Loaded dense transformer weights, executed through ITensorBackend. Currently supports the
/// bias-free mistral3 layout with full-head adjacent-pair RoPE, optional YaRN and temperature.
/// Owns uploaded weights and normalization tensors, but borrows the backend and tensor source.
/// Keep both alive until this model and all submitted operations have finished. Do not mutate
/// source descriptors or payloads while loading/using it. Concurrent calls are not supported.
[<Sealed>]
type TransformerModel private (backend: ITensorBackend, configuration: ModelConfiguration,
                               vocabularySize: int, queryWidth: int, keyValueWidth: int,
                               rope: RotaryEmbedding, embedding: IWeightTensor,
                               outputNormalization: ITensor, output: IWeightTensor,
                               layers: LoadedTransformerLayer array, storage: IDisposable array) =
    let mutable disposed = false

    let feedForward (layer: LoadedTransformerLayer) hidden (scratch: FeedForwardScratch) =
        backend.RmsNorm(hidden, layer.FeedForwardNormalization, configuration.RmsNormalizationEpsilon, scratch.Normalized)
        backend.MatMul(layer.FeedForwardGate, scratch.Normalized, scratch.Gate)
        backend.MatMul(layer.FeedForwardUp, scratch.Normalized, scratch.Up)
        backend.SwiGlu(scratch.Gate, scratch.Up, scratch.Gate)
        backend.MatMul(layer.FeedForwardDown, scratch.Gate, scratch.Residual)
        backend.Add(hidden, scratch.Residual, hidden)

    member _.Configuration = configuration
    member _.VocabularySize = vocabularySize

    member internal _.ExecutionBackend = backend

    member internal this.EnsureAlive() = ObjectDisposedException.ThrowIf(disposed, this)

    /// Runs consecutive token IDs starting at absolute position, returning caller-owned F32
    /// logits [selected tokens; VocabularySize] on the backend, without softmax or sampling.
    /// AllTokens returns every row; LastToken returns one row for a non-empty batch;
    /// NoTokens only populates the cache. Empty batches return no rows in every mode
    /// and do not truncate the cache. Omitted final feed-forward/output computations
    /// cannot affect cached keys/values. Selection never changes cache positions.
    /// The caller supplies a cache from this backend with matching layers, KV heads and head
    /// dimension, capacity at most ContextLength, and equal stored counts in every layer.
    /// Use a separate cache for each sequence/model. position may replace an existing suffix,
    /// but must not leave a gap; successful non-empty calls discard the old suffix in all layers.
    ///
    /// IDs and cache dimensions/counts are validated before any cache write. Invalid IDs, position
    /// or capacity raise ArgumentOutOfRangeException; incompatible caches raise ArgumentException.
    /// Null arguments raise ArgumentNullException; disposed storage raises ObjectDisposedException.
    /// Allocation/backend failures propagate and can leave a partially updated cache: discard it
    /// after an execution failure. No rollback is promised. Results follow backend submission
    /// ordering; Download/Synchronize waits for completion. Input IDs are consumed before return.
    member this.Forward(cache: IKeyValueCache, position: int, tokenIds: ReadOnlyMemory<int>, selection: LogitSelection) : ITensor =
        ObjectDisposedException.ThrowIf(disposed, this)
        ArgumentNullException.ThrowIfNull(cache)
        let cacheLayout = cache.Layout
        if cacheLayout.Layers <> configuration.BlockCount
           || cacheLayout.KeyValueHeads <> configuration.KeyValueHeadCount
           || cacheLayout.HeadDimension <> configuration.KeyLength
           || cacheLayout.Capacity > configuration.ContextLength then
            invalidArg (nameof cache) "The cache must match the model and fit its context length."
        let stored = cache.StoredPositions(0)
        for layer = 1 to configuration.BlockCount - 1 do
            if cache.StoredPositions(layer) <> stored then
                invalidArg (nameof cache) "Every layer must have the same stored position count."
        let tokens = tokenIds.Length
        let outputTokens =
            match selection with
            | LogitSelection.AllTokens -> tokens
            | LogitSelection.LastToken -> min 1 tokens
            | LogitSelection.NoTokens -> 0
        if position < 0 || position > stored || tokens > cacheLayout.Capacity - position then
            raise (ArgumentOutOfRangeException(nameof position, "The batch must fit the cache without leaving a gap."))
        for token in tokenIds.Span do
            if token < 0 || token >= vocabularySize then
                raise (ArgumentOutOfRangeException(nameof tokenIds, "Every token ID must belong to the vocabulary."))

        // An empty read validates cache ownership through the backend without changing the prefix.
        use emptyKeys = backend.Allocate(TensorShape.Create [| 0; keyValueWidth |])
        use emptyValues = backend.Allocate(emptyKeys.Shape)
        backend.ReadKeyValues(cache, 0, 0, emptyKeys, emptyValues)
        let logits = backend.Allocate(TensorShape.Create [| outputTokens; vocabularySize |])
        try
            if tokens > 0 then
                // Scratch is reused across layers; weights remain in their stored encoding.
                let allocate width = backend.Allocate(TensorShape.Create [| tokens; width |])
                use hidden = allocate configuration.EmbeddingLength
                use normalized = allocate configuration.EmbeddingLength
                use residual = allocate configuration.EmbeddingLength
                use queries = allocate queryWidth
                use keys = allocate keyValueWidth
                use values = allocate keyValueWidth
                use attended = allocate queryWidth
                use gate = allocate configuration.FeedForwardLength
                use up = allocate configuration.FeedForwardLength
                let scratch = { Normalized = normalized; Gate = gate; Up = up; Residual = residual }
                let finish hidden normalized =
                    backend.RmsNorm(hidden, outputNormalization, configuration.RmsNormalizationEpsilon, normalized)
                    backend.MatMul(output, normalized, logits)
                let positions = ReadOnlyMemory<int>(Array.init tokens (fun token -> position + token))
                backend.EmbeddingLookup(embedding, tokenIds, hidden)
                for index = 0 to layers.Length - 1 do
                    let layer = layers[index]
                    backend.RmsNorm(hidden, layer.AttentionNormalization, configuration.RmsNormalizationEpsilon, normalized)
                    let lastLayer = index = layers.Length - 1
                    let needsOutput = not lastLayer || outputTokens > 0
                    if needsOutput then
                        backend.MatMul(layer.Query, normalized, queries)
                        backend.Rope(queries, positions, rope, queries)
                        match configuration.AttentionTemperature with
                        | Some temperature -> backend.ApplyAttentionTemperature(queries, positions, temperature, queries)
                        | None -> ()
                    backend.MatMul(layer.Key, normalized, keys)
                    backend.MatMul(layer.Value, normalized, values)
                    backend.Rope(keys, positions, rope, keys)
                    backend.StoreKeyValues(cache, index, position, keys, values)
                    if needsOutput then
                        backend.Attention(cache, index, position, queries, attended)
                        backend.MatMul(layer.AttentionOutput, attended, residual)
                        backend.Add(hidden, residual, hidden)
                        if lastLayer && outputTokens < tokens then
                            let allocateSelected width = backend.Allocate(TensorShape.Create [| outputTokens; width |])
                            use selectedHidden = allocateSelected configuration.EmbeddingLength
                            use selectedNormalized = allocateSelected configuration.EmbeddingLength
                            use selectedResidual = allocateSelected configuration.EmbeddingLength
                            use selectedGate = allocateSelected configuration.FeedForwardLength
                            use selectedUp = allocateSelected configuration.FeedForwardLength
                            backend.CopyRows(hidden, tokens - 1, selectedHidden)
                            let selectedScratch =
                                { Normalized = selectedNormalized; Gate = selectedGate
                                  Up = selectedUp; Residual = selectedResidual }
                            feedForward layer selectedHidden selectedScratch
                            finish selectedHidden selectedNormalized
                        else
                            feedForward layer hidden scratch
                            if lastLayer then finish hidden normalized

            logits
        with _ ->
            logits.Dispose()
            reraise ()

    /// Forward pass returning logits for all tokens; equivalent to AllTokens selection.
    /// Preserves the full-output contract for scoring and reference comparisons.
    member this.Forward(cache: IKeyValueCache, position: int, tokenIds: ReadOnlyMemory<int>) : ITensor =
        this.Forward(cache, position, tokenIds, LogitSelection.AllTokens)

    /// Validates configuration, role shapes, encodings and matching source descriptors, then loads
    /// weights once. Matrices retain stored encodings; only the small normalization vectors are
    /// decoded to F32. Tied output shares the uploaded embedding. Invalid layouts/payloads raise
    /// InvalidDataException; invalid rotary parameters raise ArgumentException. I/O and backend
    /// exceptions propagate. Failed loads dispose any storage already uploaded. Neither source nor
    /// backend is disposed by this model. Dispose the model before disposing either dependency.
    static member Load(backend: ITensorBackend, source: ITensorSource, layout: ModelTensorLayout) =
        ArgumentNullException.ThrowIfNull(backend)
        let descriptors, queryWidth, keyValueWidth = ModelLoading.validate source layout
        let rope = ModelLoading.rope layout.Configuration
        let owned = ResizeArray<IDisposable>()
        let matrix (descriptor: TensorDescriptor) =
            let weight = backend.UploadWeights(WeightMatrix.Read(source, descriptor.Name))
            owned.Add weight
            weight
        let normalization (descriptor: TensorDescriptor) =
            let payload = source.ReadPayload(descriptor.Name)
            if Some (uint64 payload.Length) <> descriptor.ByteSize then
                raise (InvalidDataException($"Tensor '{descriptor.Name}' has an incorrect payload length."))
            let values = Array.zeroCreate<single> layout.Configuration.EmbeddingLength
            RowDecoder.decode (descriptor.Encoding.Value, payload.Span, values.AsSpan())
            let tensor = backend.Upload(TensorShape.Create [| values.Length |], ReadOnlyMemory<single>(values))
            owned.Add tensor
            tensor
        try
            let embedding = matrix layout.TokenEmbedding
            let output = if layout.HasTiedOutput then embedding else matrix layout.Output
            let outputNormalization = normalization layout.OutputNormalization
            let layers = descriptors |> Array.map (fun layer ->
                { AttentionNormalization = normalization layer.AttentionNormalization
                  Query = matrix layer.Query
                  Key = matrix layer.Key
                  Value = matrix layer.Value
                  AttentionOutput = matrix layer.AttentionOutput
                  FeedForwardNormalization = normalization layer.FeedForwardNormalization
                  FeedForwardGate = matrix layer.FeedForwardGate
                  FeedForwardUp = matrix layer.FeedForwardUp
                  FeedForwardDown = matrix layer.FeedForwardDown })
            new TransformerModel(backend, layout.Configuration, layout.VocabularySize, queryWidth,
                                 keyValueWidth, rope, embedding, outputNormalization, output, layers, owned.ToArray())
        with _ ->
            for resource in owned do resource.Dispose()
            reraise ()

    interface IDisposable with
        member _.Dispose() =
            if not disposed then
                disposed <- true
                for resource in storage do resource.Dispose()
