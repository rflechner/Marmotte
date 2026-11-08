namespace Marmotte.Gguf

open System
open System.IO
open Marmotte.Core

/// Metadata keys used by the model configuration reader. Architecture is a complete key;
/// other constants are suffixes prefixed with the architecture name and a dot.
/// RopeScalingPrefix identifies a diagnostic group, not a stored metadata value.
/// Format overview: https://huggingface.co/docs/hub/gguf
/// Standard key list: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md#standardized-key-value-pairs
/// Additional architecture-specific keys and references: docs/model-configuration.md.
[<RequireQualifiedAccess>]
module private ModelMetadataKeys =
    [<Literal>]
    let Architecture = "general.architecture"

    [<Literal>]
    let ContextLength = "context_length"

    [<Literal>]
    let BlockCount = "block_count"

    [<Literal>]
    let EmbeddingLength = "embedding_length"

    [<Literal>]
    let FeedForwardLength = "feed_forward_length"

    [<Literal>]
    let ExpertCount = "expert_count"

    [<Literal>]
    let ExpertUsedCount = "expert_used_count"

    [<Literal>]
    let AttentionHeadCount = "attention.head_count"

    [<Literal>]
    let AttentionKeyValueHeadCount = "attention.head_count_kv"

    [<Literal>]
    let AttentionKeyLength = "attention.key_length"

    [<Literal>]
    let AttentionValueLength = "attention.value_length"

    [<Literal>]
    let AttentionSlidingWindow = "attention.sliding_window"

    [<Literal>]
    let AttentionRmsNormalizationEpsilon = "attention.layer_norm_rms_epsilon"

    [<Literal>]
    let AttentionTemperatureScale = "attention.temperature_scale"

    [<Literal>]
    let RopeDimensionCount = "rope.dimension_count"

    [<Literal>]
    let RopeFrequencyBase = "rope.freq_base"

    [<Literal>]
    let RopeScalingType = "rope.scaling.type"

    [<Literal>]
    let RopeScalingFactor = "rope.scaling.factor"

    [<Literal>]
    let RopeScalingOriginalContextLength = "rope.scaling.original_context_length"

    [<Literal>]
    let RopeScalingAttentionFactor = "rope.scaling.attn_factor"

    [<Literal>]
    let RopeScalingYarnBetaFast = "rope.scaling.yarn_beta_fast"

    [<Literal>]
    let RopeScalingYarnBetaSlow = "rope.scaling.yarn_beta_slow"

    [<Literal>]
    let RopeScalingYarnLogMultiplier = "rope.scaling.yarn_log_multiplier"

    [<Literal>]
    let RopeScalingPrefix = "rope.scaling"

/// Interprets typed metadata without reading weights or changing the GGUF index reader.
[<RequireQualifiedAccess>]
module GgufModelConfiguration =
    let private fail key message =
        raise (InvalidDataException($"Model metadata '{key}': {message}"))

    /// Read the supported dense mistral3 hyperparameters. Required counts are UInt32 and
    /// real numbers are Float32; no numeric coercion is performed. Optional defaults and
    /// supported variants are specified in docs/model-configuration.md.
    /// Raises InvalidDataException for missing, mistyped, inconsistent or unsupported metadata,
    /// and ArgumentNullException for a null map. Does not validate tensors or the tokenizer.
    let read (metadata: Map<string, GgufValue>) : ModelConfiguration =
        ArgumentNullException.ThrowIfNull(metadata)

        let required key =
            match Map.tryFind key metadata with
            | Some value -> value
            | None -> fail key "required value is missing."

        let stringValue key value =
            match value with
            | GgufValue.String text -> text
            | _ -> fail key "expected String."

        let countValue key value =
            match value with
            | GgufValue.UInt32 count when count > 0u && count <= uint32 Int32.MaxValue -> int count
            | _ -> fail key "expected a positive UInt32 representable as Int32."

        let realValue key value =
            match value with
            | GgufValue.Float32 number when Single.IsFinite number -> double number
            | _ -> fail key "expected a finite Float32."

        let architecture = stringValue ModelMetadataKeys.Architecture (required ModelMetadataKeys.Architecture)
        if architecture <> "mistral3" then
            fail ModelMetadataKeys.Architecture $"unsupported architecture '{architecture}'."

        let key suffix = architecture + "." + suffix
        let count suffix = countValue (key suffix) (required (key suffix))
        let real suffix = realValue (key suffix) (required (key suffix))
        let optional suffix parse fallback =
            match Map.tryFind (key suffix) metadata with
            | None -> fallback
            | Some value -> parse (key suffix) value
        let optionalReal suffix fallback = optional suffix realValue fallback
        let positive suffix value =
            if value <= 0.0 then fail (key suffix) "must be positive."
            value

        // These variants need additional configuration and execution contracts.
        for suffix in [ ModelMetadataKeys.ExpertCount; ModelMetadataKeys.ExpertUsedCount; ModelMetadataKeys.AttentionSlidingWindow ] do
            match Map.tryFind (key suffix) metadata with
            | None | Some (GgufValue.UInt32 0u) -> ()
            | _ -> fail (key suffix) "only the dense, full-attention variant is supported (absent or UInt32 zero)."

        let contextLength = count ModelMetadataKeys.ContextLength
        let blockCount = count ModelMetadataKeys.BlockCount
        let embeddingLength = count ModelMetadataKeys.EmbeddingLength
        let feedForwardLength = count ModelMetadataKeys.FeedForwardLength
        let headCount = count ModelMetadataKeys.AttentionHeadCount
        let keyValueHeadCount = count ModelMetadataKeys.AttentionKeyValueHeadCount
        if headCount % keyValueHeadCount <> 0 then
            fail (key ModelMetadataKeys.AttentionKeyValueHeadCount) "must divide the query head count."

        // Explicit head lengths matter: the target's query width differs from its embedding width.
        let keyLength = count ModelMetadataKeys.AttentionKeyLength
        let valueLength = count ModelMetadataKeys.AttentionValueLength
        let rotaryLength = optional ModelMetadataKeys.RopeDimensionCount countValue keyLength
        if keyLength % 2 <> 0 || rotaryLength <> keyLength then
            fail (key ModelMetadataKeys.RopeDimensionCount) "only full-head RoPE with an even key length is supported."
        for suffix, heads, width in
            [ ModelMetadataKeys.AttentionKeyLength, headCount, keyLength
              ModelMetadataKeys.AttentionValueLength, keyValueHeadCount, valueLength ] do
            if int64 heads * int64 width > int64 Int32.MaxValue then
                fail (key suffix) "the projected width exceeds Int32 capacity."

        let epsilon = real ModelMetadataKeys.AttentionRmsNormalizationEpsilon |> positive ModelMetadataKeys.AttentionRmsNormalizationEpsilon
        let frequencyBase = real ModelMetadataKeys.RopeFrequencyBase |> positive ModelMetadataKeys.RopeFrequencyBase
        let scalingType = optional ModelMetadataKeys.RopeScalingType stringValue "none"
        let originalContext =
            match Map.tryFind (key ModelMetadataKeys.RopeScalingOriginalContextLength) metadata with
            | None -> None
            | Some value -> Some (countValue (key ModelMetadataKeys.RopeScalingOriginalContextLength) value)
        let requireOriginalContext () =
            match originalContext with
            | Some length when length <= contextLength -> length
            | Some _ -> fail (key ModelMetadataKeys.RopeScalingOriginalContextLength) "exceeds the model context length."
            | None -> fail (key ModelMetadataKeys.RopeScalingOriginalContextLength) "required for YaRN or attention temperature."

        let scaling =
            match scalingType with
            | "none" ->
                for suffix in
                    [ ModelMetadataKeys.RopeScalingFactor; ModelMetadataKeys.RopeScalingAttentionFactor; ModelMetadataKeys.RopeScalingYarnBetaFast
                      ModelMetadataKeys.RopeScalingYarnBetaSlow; ModelMetadataKeys.RopeScalingYarnLogMultiplier ] do
                    if Map.containsKey (key suffix) metadata then
                        fail (key suffix) "requires YaRN scaling."
                ModelRopeScaling.None
            | "yarn" ->
                let parameters =
                    { ScalingFactor = real ModelMetadataKeys.RopeScalingFactor
                      OriginalContextLength = requireOriginalContext ()
                      BetaFast = optionalReal ModelMetadataKeys.RopeScalingYarnBetaFast 32.0
                      BetaSlow = optionalReal ModelMetadataKeys.RopeScalingYarnBetaSlow 1.0
                      LogMultiplier = optionalReal ModelMetadataKeys.RopeScalingYarnLogMultiplier 0.0
                      AttentionFactor = optionalReal ModelMetadataKeys.RopeScalingAttentionFactor 1.0 }
                if frequencyBase <= 1.0 then
                    fail (key ModelMetadataKeys.RopeFrequencyBase) "YaRN requires a base greater than one."
                if parameters.ScalingFactor < 1.0 then
                    fail (key ModelMetadataKeys.RopeScalingFactor) "must be at least one."
                if parameters.BetaSlow <= 0.0 || parameters.BetaFast < parameters.BetaSlow then
                    fail (key ModelMetadataKeys.RopeScalingYarnBetaFast) "requires beta_fast >= beta_slow > 0."
                if parameters.LogMultiplier < 0.0 || parameters.AttentionFactor <= 0.0 then
                    fail (key ModelMetadataKeys.RopeScalingPrefix) "requires a non-negative log multiplier and positive attention factor."
                let logarithmicScale = 0.1 * Math.Log parameters.ScalingFactor
                let magnitude =
                    single (parameters.AttentionFactor * (1.0 + logarithmicScale)
                            / (1.0 + parameters.LogMultiplier * logarithmicScale))
                if not (Single.IsFinite magnitude) || magnitude <= 0.0f then
                    fail (key ModelMetadataKeys.RopeScalingPrefix) "effective magnitude must be finite and positive in F32."
                ModelRopeScaling.Yarn parameters
            | other -> fail (key ModelMetadataKeys.RopeScalingType) $"unsupported scaling '{other}'."

        let temperatureScale = optionalReal ModelMetadataKeys.AttentionTemperatureScale 0.0
        if temperatureScale < 0.0 then
            fail (key ModelMetadataKeys.AttentionTemperatureScale) "must be non-negative."
        let temperature =
            if temperatureScale = 0.0 then None
            else Some (AttentionTemperature.Create(temperatureScale, requireOriginalContext ()))

        { Architecture = ModelArchitecture.Mistral3
          ContextLength = contextLength
          BlockCount = blockCount
          EmbeddingLength = embeddingLength
          FeedForwardLength = feedForwardLength
          AttentionHeadCount = headCount
          KeyValueHeadCount = keyValueHeadCount
          KeyLength = keyLength
          ValueLength = valueLength
          RmsNormalizationEpsilon = single epsilon
          RopeFrequencyBase = frequencyBase
          RopeScaling = scaling
          AttentionTemperature = temperature }
