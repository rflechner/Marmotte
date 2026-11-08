namespace Marmotte.Gguf

open System
open System.IO
open Marmotte.Core

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

        let architecture = stringValue "general.architecture" (required "general.architecture")
        if architecture <> "mistral3" then
            fail "general.architecture" $"unsupported architecture '{architecture}'."

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
        for suffix in [ "expert_count"; "expert_used_count"; "attention.sliding_window" ] do
            match Map.tryFind (key suffix) metadata with
            | None | Some (GgufValue.UInt32 0u) -> ()
            | _ -> fail (key suffix) "only the dense, full-attention variant is supported (absent or UInt32 zero)."

        let contextLength = count "context_length"
        let blockCount = count "block_count"
        let embeddingLength = count "embedding_length"
        let feedForwardLength = count "feed_forward_length"
        let headCount = count "attention.head_count"
        let keyValueHeadCount = count "attention.head_count_kv"
        if headCount % keyValueHeadCount <> 0 then
            fail (key "attention.head_count_kv") "must divide the query head count."

        // Explicit head lengths matter: the target's query width differs from its embedding width.
        let keyLength = count "attention.key_length"
        let valueLength = count "attention.value_length"
        let rotaryLength = optional "rope.dimension_count" countValue keyLength
        if keyLength % 2 <> 0 || rotaryLength <> keyLength then
            fail (key "rope.dimension_count") "only full-head RoPE with an even key length is supported."
        for suffix, heads, width in
            [ "attention.key_length", headCount, keyLength
              "attention.value_length", keyValueHeadCount, valueLength ] do
            if int64 heads * int64 width > int64 Int32.MaxValue then
                fail (key suffix) "the projected width exceeds Int32 capacity."

        let epsilon = real "attention.layer_norm_rms_epsilon" |> positive "attention.layer_norm_rms_epsilon"
        let frequencyBase = real "rope.freq_base" |> positive "rope.freq_base"
        let scalingType = optional "rope.scaling.type" stringValue "none"
        let originalContext =
            match Map.tryFind (key "rope.scaling.original_context_length") metadata with
            | None -> None
            | Some value -> Some (countValue (key "rope.scaling.original_context_length") value)
        let requireOriginalContext () =
            match originalContext with
            | Some length when length <= contextLength -> length
            | Some _ -> fail (key "rope.scaling.original_context_length") "exceeds the model context length."
            | None -> fail (key "rope.scaling.original_context_length") "required for YaRN or attention temperature."

        let scaling =
            match scalingType with
            | "none" ->
                for suffix in
                    [ "rope.scaling.factor"; "rope.scaling.attn_factor"; "rope.scaling.yarn_beta_fast"
                      "rope.scaling.yarn_beta_slow"; "rope.scaling.yarn_log_multiplier" ] do
                    if Map.containsKey (key suffix) metadata then
                        fail (key suffix) "requires YaRN scaling."
                ModelRopeScaling.None
            | "yarn" ->
                let parameters =
                    { ScalingFactor = real "rope.scaling.factor"
                      OriginalContextLength = requireOriginalContext ()
                      BetaFast = optionalReal "rope.scaling.yarn_beta_fast" 32.0
                      BetaSlow = optionalReal "rope.scaling.yarn_beta_slow" 1.0
                      LogMultiplier = optionalReal "rope.scaling.yarn_log_multiplier" 0.0
                      AttentionFactor = optionalReal "rope.scaling.attn_factor" 1.0 }
                if frequencyBase <= 1.0 then
                    fail (key "rope.freq_base") "YaRN requires a base greater than one."
                if parameters.ScalingFactor < 1.0 then
                    fail (key "rope.scaling.factor") "must be at least one."
                if parameters.BetaSlow <= 0.0 || parameters.BetaFast < parameters.BetaSlow then
                    fail (key "rope.scaling.yarn_beta_fast") "requires beta_fast >= beta_slow > 0."
                if parameters.LogMultiplier < 0.0 || parameters.AttentionFactor <= 0.0 then
                    fail (key "rope.scaling") "requires a non-negative log multiplier and positive attention factor."
                let logarithmicScale = 0.1 * Math.Log parameters.ScalingFactor
                let magnitude =
                    single (parameters.AttentionFactor * (1.0 + logarithmicScale)
                            / (1.0 + parameters.LogMultiplier * logarithmicScale))
                if not (Single.IsFinite magnitude) || magnitude <= 0.0f then
                    fail (key "rope.scaling") "effective magnitude must be finite and positive in F32."
                ModelRopeScaling.Yarn parameters
            | other -> fail (key "rope.scaling.type") $"unsupported scaling '{other}'."

        let temperatureScale = optionalReal "attention.temperature_scale" 0.0
        if temperatureScale < 0.0 then
            fail (key "attention.temperature_scale") "must be non-negative."
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
