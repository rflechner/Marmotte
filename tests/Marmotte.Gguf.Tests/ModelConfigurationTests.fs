namespace Marmotte.Gguf

open System
open System.IO
open Marmotte.Core
open Marmotte.Gguf
open Xunit

module ModelConfigurationTests =
    let private target =
        [ "general.architecture", GgufValue.String "mistral3"
          "mistral3.context_length", GgufValue.UInt32 262144u
          "mistral3.block_count", GgufValue.UInt32 26u
          "mistral3.embedding_length", GgufValue.UInt32 3072u
          "mistral3.feed_forward_length", GgufValue.UInt32 9216u
          "mistral3.attention.head_count", GgufValue.UInt32 32u
          "mistral3.attention.head_count_kv", GgufValue.UInt32 8u
          "mistral3.attention.key_length", GgufValue.UInt32 128u
          "mistral3.attention.value_length", GgufValue.UInt32 128u
          "mistral3.attention.layer_norm_rms_epsilon", GgufValue.Float32 0.00001f
          "mistral3.rope.freq_base", GgufValue.Float32 1000000.0f
          "mistral3.rope.scaling.type", GgufValue.String "yarn"
          "mistral3.rope.scaling.factor", GgufValue.Float32 16.0f
          "mistral3.rope.scaling.original_context_length", GgufValue.UInt32 16384u
          "mistral3.rope.scaling.yarn_log_multiplier", GgufValue.Float32 1.0f
          "mistral3.attention.temperature_scale", GgufValue.Float32 0.1f ]
        |> Map.ofList

    [<Fact>]
    let ``Target metadata preserves independent head and embedding widths`` () =
        let config = GgufModelConfiguration.read target
        Assert.Equal(ModelArchitecture.Mistral3, config.Architecture)
        Assert.Equal(26, config.BlockCount)
        Assert.Equal(3072, config.EmbeddingLength)
        Assert.Equal(9216, config.FeedForwardLength)
        Assert.Equal(262144, config.ContextLength)
        Assert.Equal(4096, config.AttentionHeadCount * config.KeyLength)
        Assert.Equal(1024, config.KeyValueHeadCount * config.ValueLength)
        Assert.Equal(0.00001f, config.RmsNormalizationEpsilon)
        Assert.Equal(1.0f, config.AttentionTemperature.Value.Factor(16383))
        Assert.True(config.AttentionTemperature.Value.Factor(16384) > 1.0f)
        match config.RopeScaling with
        | ModelRopeScaling.Yarn parameters ->
            Assert.Equal(16.0, parameters.ScalingFactor)
            Assert.Equal(16384, parameters.OriginalContextLength)
            Assert.Equal(32.0, parameters.BetaFast)
            Assert.Equal(1.0, parameters.BetaSlow)
            let rope = RotaryEmbedding.Yarn(config.KeyLength, config.RopeFrequencyBase, RopePairing.AdjacentPairs, parameters)
            Assert.Equal(1.0f, rope.Magnitude)
        | _ -> failwith "Expected YaRN."

    [<Fact>]
    let ``Plain rotary configuration uses documented optional defaults`` () =
        let metadata =
            target
            |> Map.filter (fun key _ -> not (key.Contains("rope.scaling.")) && not (key.EndsWith("temperature_scale")))
        let config = GgufModelConfiguration.read metadata
        Assert.Equal(ModelRopeScaling.None, config.RopeScaling)
        Assert.True(config.AttentionTemperature.IsNone)

    [<Fact>]
    let ``Every required value is checked for absence and exact type`` () =
        for key, _ in Map.toList target do
            if not (key.EndsWith("yarn_log_multiplier") || key.EndsWith("temperature_scale") || key.EndsWith("scaling.type")) then
                let error = Assert.Throws<InvalidDataException>(fun () -> target |> Map.remove key |> GgufModelConfiguration.read |> ignore)
                Assert.Contains(key, error.Message)
            let error = Assert.Throws<InvalidDataException>(fun () -> target |> Map.add key (GgufValue.Bool true) |> GgufModelConfiguration.read |> ignore)
            Assert.Contains(key, error.Message)

    [<Fact>]
    let ``Invalid dimensions scalars and unsupported variants are rejected`` () =
        let invalidValues =
            [ "general.architecture", GgufValue.String "unknown"
              "mistral3.block_count", GgufValue.UInt32 0u
              "mistral3.context_length", GgufValue.UInt32 UInt32.MaxValue
              "mistral3.embedding_length", GgufValue.Int32 3072
              "mistral3.attention.head_count_kv", GgufValue.UInt32 7u
              "mistral3.attention.head_count", GgufValue.UInt32 2147483640u
              "mistral3.attention.key_length", GgufValue.UInt32 127u
              "mistral3.rope.dimension_count", GgufValue.UInt32 64u
              "mistral3.attention.layer_norm_rms_epsilon", GgufValue.Float32 0.0f
              "mistral3.rope.freq_base", GgufValue.Float32 Single.NaN
              "mistral3.rope.freq_base", GgufValue.Float32 1.0f
              "mistral3.rope.scaling.factor", GgufValue.Float32 0.5f
              "mistral3.rope.scaling.factor", GgufValue.Float32 Single.PositiveInfinity
              "mistral3.rope.scaling.original_context_length", GgufValue.UInt32 262145u
              "mistral3.rope.scaling.yarn_beta_slow", GgufValue.Float32 0.0f
              "mistral3.rope.scaling.yarn_beta_fast", GgufValue.Float32 0.5f
              "mistral3.rope.scaling.yarn_log_multiplier", GgufValue.Float32 -1.0f
              "mistral3.rope.scaling.attn_factor", GgufValue.Float32 0.0f
              "mistral3.rope.scaling.type", GgufValue.String "linear"
              "mistral3.rope.scaling.type", GgufValue.String "none"
              "mistral3.attention.temperature_scale", GgufValue.Float32 -0.1f
              "mistral3.expert_count", GgufValue.UInt32 8u
              "mistral3.attention.sliding_window", GgufValue.UInt32 4096u ]
        for key, value in invalidValues do
            Assert.Throws<InvalidDataException>(fun () -> target |> Map.add key value |> GgufModelConfiguration.read |> ignore) |> ignore

    [<Fact>]
    let ``Temperature without YaRN still requires an original context`` () =
        let metadata = target |> Map.filter (fun key _ -> not (key.Contains("rope.scaling.")))
        Assert.Throws<InvalidDataException>(fun () -> GgufModelConfiguration.read metadata |> ignore) |> ignore
        let config = metadata |> Map.add "mistral3.rope.scaling.original_context_length" (GgufValue.UInt32 16384u) |> GgufModelConfiguration.read
        Assert.True(config.AttentionTemperature.IsSome)
