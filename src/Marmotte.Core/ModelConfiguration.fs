namespace Marmotte.Core

/// Architectures whose hyperparameters Marmotte can interpret.
[<RequireQualifiedAccess>]
type ModelArchitecture =
    | Mistral3

/// Full-head rotary position encoding; all frequencies use adjacent value pairs.
[<RequireQualifiedAccess>]
type ModelRopeScaling =
    | None
    | Yarn of YarnParameters

/// Format-neutral hyperparameters, with counts measured in elements or token positions.
/// This record describes metadata only: tensor names, shapes and tokenizer validity are
/// checked separately. Public record construction does not validate its fields.
type ModelConfiguration =
    { Architecture: ModelArchitecture
      ContextLength: int
      BlockCount: int
      EmbeddingLength: int
      FeedForwardLength: int
      AttentionHeadCount: int
      KeyValueHeadCount: int
      KeyLength: int
      ValueLength: int
      RmsNormalizationEpsilon: single
      RopeFrequencyBase: double
      RopeScaling: ModelRopeScaling
      AttentionTemperature: AttentionTemperature option }
