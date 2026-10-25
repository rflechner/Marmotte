namespace Marmotte.Core

open System
open System.Collections.Generic

/// Which two values of a head rotary position embedding rotates together. Pairings and rotation
/// formulas follow the reference implementation cited in docs/kernels.md (checked 2026-10-25).
[<RequireQualifiedAccess>]
type RopePairing =
    /// Values 2j and 2j + 1: the "normal" pairing of mistral3 and other LLaMA-family
    /// architectures stored in GGUF.
    | AdjacentPairs
    /// Values j and j + HeadDimension / 2: the GPT-NeoX pairing, and the rotate_half convention
    /// of Hugging Face checkpoints.
    | SplitHalves

/// Model parameters for full-head YaRN context extension, independent of any file format.
/// RotaryEmbedding.Yarn validates the record. Positions remain absolute token indices.
type YarnParameters =
    { /// Context extension ratio, at least 1. Interpolated pairs divide their frequency by it.
      ScalingFactor: double
      /// Number of positions in the original training context, before extension; must be positive.
      OriginalContextLength: int
      /// Rotations within the original context at the fast end of the transition; usually 32.
      /// Must be finite and at least BetaSlow.
      BetaFast: double
      /// Rotations within the original context at the slow end; usually 1, finite and positive.
      BetaSlow: double
      /// Non-negative coefficient from rope.scaling.yarn_log_multiplier. Use 0 when absent
      /// for the reference YaRN magnitude, or 1 for the target mistral3 model's cancellation.
      LogMultiplier: double
      /// Additional positive multiplier of Q and K magnitudes (rope.scaling.attn_factor).
      /// Use 1 when absent. This is not the position-dependent attention.temperature_scale.
      AttentionFactor: double }

module private YarnPreparation =
    let validate (parameters: YarnParameters) =
        ArgumentNullException.ThrowIfNull(parameters)

        let requireFinite name value predicate message =
            if not (Double.IsFinite value) || not (predicate value) then
                raise (ArgumentOutOfRangeException(name, box value, message))

        requireFinite "ScalingFactor" parameters.ScalingFactor (fun value -> value >= 1.0)
            "The YaRN scaling factor must be finite and at least one."
        if parameters.OriginalContextLength <= 0 then
            invalidArg "OriginalContextLength" "The original context length must be positive."
        requireFinite "BetaSlow" parameters.BetaSlow (fun value -> value > 0.0)
            "The slow rotation threshold must be finite and positive."
        requireFinite "BetaFast" parameters.BetaFast (fun value -> value >= parameters.BetaSlow)
            "The fast rotation threshold must be finite and at least the slow threshold."
        requireFinite "LogMultiplier" parameters.LogMultiplier (fun value -> value >= 0.0)
            "The YaRN logarithmic multiplier must be finite and non-negative."
        requireFinite "AttentionFactor" parameters.AttentionFactor (fun value -> value > 0.0)
            "The attention factor must be finite and positive."

    let magnitude (parameters: YarnParameters) =
        // Combine the reference's context compensation and RoPE magnitude once, before rotation.
        // No architecture-specific policy (such as DeepSeek's numerator adjustment) is inferred.
        let logarithmicScale = 0.1 * Math.Log parameters.ScalingFactor
        let correction = (1.0 + logarithmicScale) / (1.0 + parameters.LogMultiplier * logarithmicScale)
        let result = single (parameters.AttentionFactor * correction)
        if not (Single.IsFinite result) || result <= 0.0f then
            invalidArg "parameters" "The effective YaRN magnitude must be representable as a finite, positive F32 value."
        result

    let frequencies headDimension frequencyBase (parameters: YarnParameters) =
        // A pair makes L * base^(-2j/D) / (2*pi) rotations in L positions. Solve for j.
        // Separate logarithms avoid overflow/underflow for finite but extreme thresholds.
        let contextLog = Math.Log(float parameters.OriginalContextLength) - Math.Log(Math.Tau)
        let pairAtRotations rotations =
            float headDimension / (2.0 * Math.Log frequencyBase) * (contextLog - Math.Log rotations)

        // The reference bounds the upper cutoff by D - 1, not by the number of pairs minus one.
        let fastPair = max 0.0 (Math.Floor(pairAtRotations parameters.BetaFast))
        let slowPair = min (float (headDimension - 1)) (Math.Ceiling(pairAtRotations parameters.BetaSlow))
        let transitionWidth = max 0.001 (slowPair - fastPair)

        Array.init (headDimension / 2) (fun pair ->
            let frequency = Math.Pow(frequencyBase, -2.0 * float pair / float headDimension)
            let interpolation = Math.Clamp((float pair - fastPair) / transitionWidth, 0.0, 1.0)
            frequency * (1.0 - interpolation) + (frequency / parameters.ScalingFactor) * interpolation)

/// Rotary position embedding (RoPE) parameters of one model: one angular frequency per pair of
/// head values, a magnitude factor and a pairing. At position p, pair j rotates by the angle
/// p × Frequencies[j], and both rotated values are multiplied by Magnitude. Frequencies stay in
/// double precision so that angles remain accurate at long positions; the reference
/// implementation accumulates them in F32. Every value of a head is rotated: partial rotary
/// dimensions are not supported.
[<Sealed>]
type RotaryEmbedding private (frequencies: double array, magnitude: single, pairing: RopePairing) =
    let readOnlyFrequencies = Array.AsReadOnly(frequencies)

    /// Angular frequency of each pair, in radians per position.
    member _.Frequencies = readOnlyFrequencies :> IReadOnlyList<double>

    /// Values per head: two per frequency.
    member _.HeadDimension = 2 * frequencies.Length

    /// Factor applied to both rotated values: 1 for plain RoPE; YaRN corrects attention with it.
    member _.Magnitude = magnitude

    member _.Pairing = pairing

    /// Plain RoPE, as in "RoFormer" (https://arxiv.org/abs/2104.09864): pair j of a head of
    /// headDimension values rotates by frequencyBase^(-2j / headDimension) radians per position,
    /// with magnitude 1. The target model has 128 values per head and rope.freq_base 1,000,000.
    /// Raises ArgumentException for an odd or non-positive head dimension, and
    /// ArgumentOutOfRangeException for a base that is not finite and positive.
    static member Standard(headDimension: int, frequencyBase: double, pairing: RopePairing) =
        if headDimension <= 0 || headDimension % 2 <> 0 then
            invalidArg (nameof headDimension) "The head dimension must be positive and even."
        if not (Double.IsFinite frequencyBase) || frequencyBase <= 0.0 then
            raise (ArgumentOutOfRangeException(nameof frequencyBase, box frequencyBase, "The frequency base must be finite and positive."))

        let frequencyOf pair = Math.Pow(frequencyBase, -2.0 * float pair / float headDimension)
        RotaryEmbedding(Array.init (headDimension / 2) frequencyOf, 1.0f, pairing)

    /// Prepare YaRN frequencies in double precision and the final F32 magnitude for IRopeKernel.
    /// Uses the full YaRN blend (an extrapolation factor of 1) and the general magnitude with
    /// mscale = 1: AttentionFactor * (1 + 0.1 ln(s)) / (1 + 0.1 LogMultiplier ln(s)).
    /// A LogMultiplier of 1 cancels the logarithmic correction; 0 gives the original YaRN rule.
    /// Apply the returned embedding to Q and K, not V. This does not load model metadata or apply
    /// the separate, position-dependent attention temperature. Pairing affects layout only.
    /// Formulas verified against the reference revision cited in docs/kernels.md, for the
    /// correction dimensions, the rotation and the magnitude compensation; see there for equations
    /// and links.
    /// Raises ArgumentException for non-positive or odd head dimensions, non-positive original
    /// context lengths or an unrepresentable magnitude; ArgumentOutOfRangeException for invalid
    /// numeric parameters, including a frequency base that is not finite and greater than 1;
    /// and ArgumentNullException for null parameters. Context contraction is not supported.
    static member Yarn(headDimension: int, frequencyBase: double, pairing: RopePairing, parameters: YarnParameters) =
        if headDimension <= 0 || headDimension % 2 <> 0 then
            invalidArg (nameof headDimension) "The head dimension must be positive and even."
        if not (Double.IsFinite frequencyBase) || frequencyBase <= 1.0 then
            raise (ArgumentOutOfRangeException(nameof frequencyBase, box frequencyBase, "The YaRN frequency base must be finite and greater than one."))
        YarnPreparation.validate parameters

        let magnitude = YarnPreparation.magnitude parameters
        let frequencies = YarnPreparation.frequencies headDimension frequencyBase parameters
        RotaryEmbedding(frequencies, magnitude, pairing)

    /// Custom frequencies, such as YaRN-corrected ones or ones divided by a model's rope_freqs
    /// factors; the array is copied. Raises ArgumentException for an empty array or a frequency
    /// that is not finite and non-negative, and ArgumentOutOfRangeException for a magnitude that
    /// is not finite and positive.
    static member Create(frequencies: double array, magnitude: single, pairing: RopePairing) =
        ArgumentNullException.ThrowIfNull(frequencies)
        if frequencies.Length = 0 then
            invalidArg (nameof frequencies) "Rotary embedding needs at least one frequency."
        if frequencies |> Array.exists (fun frequency -> not (Double.IsFinite frequency) || frequency < 0.0) then
            invalidArg (nameof frequencies) "Rotary frequencies must be finite and non-negative."
        if not (Single.IsFinite magnitude) || magnitude <= 0.0f then
            raise (ArgumentOutOfRangeException(nameof magnitude, box magnitude, "The magnitude must be finite and positive."))

        RotaryEmbedding(Array.copy frequencies, magnitude, pairing)
