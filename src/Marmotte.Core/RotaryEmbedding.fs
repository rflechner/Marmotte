namespace Marmotte.Core

open System
open System.Collections.Generic

/// Which two values of a head rotary position embedding rotates together. Pairings and rotation
/// formulas follow rotate_pairs in llama.cpp's ggml/src/ggml-cpu/ops.cpp (checked 2026-10-25).
[<RequireQualifiedAccess>]
type RopePairing =
    /// Values 2j and 2j + 1: llama.cpp's "normal" type (LLAMA_ROPE_TYPE_NORM), which its graph
    /// uses for mistral3 and other LLaMA-family architectures stored in GGUF.
    | AdjacentPairs
    /// Values j and j + HeadDimension / 2: llama.cpp's NeoX type, and the rotate_half convention
    /// of Hugging Face checkpoints.
    | SplitHalves

/// Rotary position embedding (RoPE) parameters of one model: one angular frequency per pair of
/// head values, a magnitude factor and a pairing. At position p, pair j rotates by the angle
/// p × Frequencies[j], and both rotated values are multiplied by Magnitude. Frequencies stay in
/// double precision so that angles remain accurate at long positions; llama.cpp accumulates them
/// in F32. Every value of a head is rotated: partial rotary dimensions are not supported.
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
