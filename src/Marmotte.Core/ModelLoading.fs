namespace Marmotte.Core

open System
open System.IO

/// Backend weights for a dense block; independent of file-format names.
type internal LoadedTransformerLayer =
    { AttentionNormalization: ITensor
      Query: IWeightTensor
      Key: IWeightTensor
      Value: IWeightTensor
      AttentionOutput: IWeightTensor
      FeedForwardNormalization: ITensor
      FeedForwardGate: IWeightTensor
      FeedForwardUp: IWeightTensor
      FeedForwardDown: IWeightTensor }

module internal ModelLoading =
    let private invalid message = raise (InvalidDataException(message))

    /// Public layouts may be constructed without a format reader: validate them before loading.
    let validate (source: ITensorSource) (layout: ModelTensorLayout) =
        ArgumentNullException.ThrowIfNull(source)
        ArgumentNullException.ThrowIfNull(layout)
        let configuration = layout.Configuration
        ArgumentNullException.ThrowIfNull(configuration)
        ArgumentNullException.ThrowIfNull(layout.Layers)
        for name, count in
            [ "ContextLength", configuration.ContextLength
              "BlockCount", configuration.BlockCount
              "EmbeddingLength", configuration.EmbeddingLength
              "FeedForwardLength", configuration.FeedForwardLength
              "AttentionHeadCount", configuration.AttentionHeadCount
              "KeyValueHeadCount", configuration.KeyValueHeadCount
              "KeyLength", configuration.KeyLength
              "VocabularySize", layout.VocabularySize ] do
            if count <= 0 then invalid $"{name} must be positive."
        if configuration.KeyLength <> configuration.ValueLength || configuration.KeyLength % 2 <> 0 then
            invalid "Dense attention requires equal, positive, even key and value head lengths."
        if configuration.AttentionHeadCount % configuration.KeyValueHeadCount <> 0 then
            invalid "Query heads must be divisible by KV heads."
        if not (Single.IsFinite configuration.RmsNormalizationEpsilon) || configuration.RmsNormalizationEpsilon <= 0.0f then
            invalid "RMS normalization epsilon must be finite and positive."
        if layout.Layers.Count <> configuration.BlockCount then
            invalid "The tensor layout must contain exactly BlockCount layers."

        match configuration.AttentionTemperature with
        | Some temperature -> ArgumentNullException.ThrowIfNull(temperature)
        | None -> ()

        let width heads =
            let width = int64 heads * int64 configuration.KeyLength
            if width > int64 Array.MaxLength then invalid "Attention width exceeds the execution tensor limit."
            int width
        let queryWidth = width configuration.AttentionHeadCount
        let keyValueWidth = width configuration.KeyValueHeadCount
        let embedding = configuration.EmbeddingLength
        let feedForward = configuration.FeedForwardLength
        let layers = layout.Layers |> Seq.toArray

        let check (expected: int array) (descriptor: TensorDescriptor) =
            ArgumentNullException.ThrowIfNull(descriptor)
            if String.IsNullOrEmpty descriptor.Name then invalid "A weight name must not be empty."
            if descriptor.Shape <> Array.map uint64 expected then
                invalid $"Tensor '{descriptor.Name}' does not have the required shape."
            let encoding =
                match descriptor.Encoding with
                | Some encoding -> encoding
                | None -> invalid $"Tensor '{descriptor.Name}' has no supported runtime encoding."
            let columns = Array.last expected
            if columns % encoding.BlockElements <> 0 then
                invalid $"Tensor '{descriptor.Name}' must contain whole encoding blocks per row."
            let elements = expected |> Array.fold (fun count dimension -> count * uint64 dimension) 1UL
            let bytes = elements / uint64 encoding.BlockElements * uint64 encoding.BlockBytes
            if bytes > uint64 Array.MaxLength then invalid $"Tensor '{descriptor.Name}' exceeds the payload limit."
            if descriptor.ElementCount <> elements || descriptor.ByteSize <> Some bytes then
                invalid $"Tensor '{descriptor.Name}' has inconsistent element or byte counts."
            match source.TryFindTensor descriptor.Name with
            | Some actual when actual = descriptor -> ()
            | _ -> invalid $"Tensor '{descriptor.Name}' does not match the source descriptor."

        check [| layout.VocabularySize; embedding |] layout.TokenEmbedding
        check [| embedding |] layout.OutputNormalization
        check [| layout.VocabularySize; embedding |] layout.Output
        if layout.HasTiedOutput && layout.Output <> layout.TokenEmbedding then
            invalid "Tied output must reuse the embedding descriptor."
        for layer in layers do
            ArgumentNullException.ThrowIfNull(layer)
            check [| embedding |] layer.AttentionNormalization
            check [| queryWidth; embedding |] layer.Query
            check [| keyValueWidth; embedding |] layer.Key
            check [| keyValueWidth; embedding |] layer.Value
            check [| embedding; queryWidth |] layer.AttentionOutput
            check [| embedding |] layer.FeedForwardNormalization
            check [| feedForward; embedding |] layer.FeedForwardGate
            check [| feedForward; embedding |] layer.FeedForwardUp
            check [| embedding; feedForward |] layer.FeedForwardDown
        layers, queryWidth, keyValueWidth

    let rope (configuration: ModelConfiguration) =
        match configuration.Architecture, configuration.RopeScaling with
        | ModelArchitecture.Mistral3, ModelRopeScaling.None ->
            RotaryEmbedding.Standard(configuration.KeyLength, configuration.RopeFrequencyBase, RopePairing.AdjacentPairs)
        | ModelArchitecture.Mistral3, ModelRopeScaling.Yarn parameters ->
            RotaryEmbedding.Yarn(configuration.KeyLength, configuration.RopeFrequencyBase, RopePairing.AdjacentPairs, parameters)
