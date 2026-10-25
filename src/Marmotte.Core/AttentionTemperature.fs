namespace Marmotte.Core

open System

/// Position-dependent query scaling that mistral3 applies after RoPE, from the GGUF keys
/// attention.temperature_scale (Scale) and rope.scaling.original_context_length (FloorLength).
/// A query at position p is multiplied by 1 + Scale × ln(floor(p / FloorLength) + 1): exactly 1
/// within the first FloorLength positions, then larger for every further FloorLength positions,
/// which multiplies attention scores by the same factor and sharpens their softmax. The formula
/// follows the reference implementation cited in docs/kernels.md (checked 2026-10-25), with the
/// zero position offset used for mistral3; its power variant is not supported. This is unrelated
/// to the sampling temperature of generated tokens.
[<Sealed>]
type AttentionTemperature private (scale: double, floorLength: int) =
    /// β, the logarithmic coefficient: 0.1 for the target model. 0 makes every factor 1.
    member _.Scale = scale

    /// L, the number of positions that share one factor: 16,384 for the target model.
    member _.FloorLength = floorLength

    /// The query factor at an absolute position, at least 0, rounded once to F32. The floor is an
    /// exact integer division, and the logarithm is computed in double precision.
    /// Raises ArgumentOutOfRangeException for a negative position.
    member _.Factor(position: int) : single =
        if position < 0 then
            raise (ArgumentOutOfRangeException(nameof position, box position, "The position must be at least 0."))
        float32 (1.0 + scale * Math.Log(float (position / floorLength) + 1.0))

    /// Raises ArgumentOutOfRangeException for a scale that is not finite and non-negative, or a
    /// floor length that is not positive. Pass the F32 metadata value converted to double, such as
    /// float 0.1f, to evaluate the same double-precision expression as the reference.
    static member Create(scale: double, floorLength: int) =
        if not (Double.IsFinite scale) || scale < 0.0 then
            raise (ArgumentOutOfRangeException(nameof scale, box scale, "The temperature scale must be finite and non-negative."))
        if floorLength <= 0 then
            raise (ArgumentOutOfRangeException(nameof floorLength, box floorLength, "The floor length must be positive."))
        AttentionTemperature(scale, floorLength)
