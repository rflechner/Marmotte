namespace Marmotte.Cpu

open System

/// Transient Q8_K activations: signed quants promoted to Int16 for safe SIMD products,
/// one F32 scale per 256 values and one integer sum per 16 values. Not a file encoding.
type internal CpuQ8Activations =
    { Quants: int16 array
      Scales: single array
      Sums: int array
      ValidTokens: bool array }

module internal CpuQ8Activations =
    /// Quantize whole 256-value blocks once for the whole input batch. The first largest
    /// magnitude chooses the sign of the scale. Nearest-even rounding uses an F32 reciprocal.
    /// Mark tokens with non-finite input or an unrepresentable reciprocal/scale for
    /// decoded-F32 fallback. See the Q8_K contract in docs/kernels.md.
    let create columns (input: ReadOnlySpan<single>) =
        if columns <= 0 || columns % 256 <> 0 || input.Length % columns <> 0 then
            invalidArg (nameof input) "Activation rows must contain whole 256-value blocks."
        let result =
            { Quants = Array.zeroCreate input.Length
              Scales = Array.zeroCreate (input.Length / 256)
              Sums = Array.zeroCreate (input.Length / 16)
              ValidTokens = Array.create (input.Length / columns) true }
        for block = 0 to result.Scales.Length - 1 do
            let mutable valid = true
            let values = input.Slice(block * 256, 256)
            let mutable extreme = 0.0f
            for index = 0 to 255 do
                let value = values[index]
                if not (Single.IsFinite value) then valid <- false
                if abs value > abs extreme then extreme <- value
            if valid && extreme <> 0.0f then
                let inverse = -127.0f / extreme
                let scale = 1.0f / inverse
                if not (Single.IsFinite inverse && Single.IsFinite scale) || scale = 0.0f then
                    valid <- false
                else
                    result.Scales[block] <- scale
                    for group = 0 to 15 do
                        let mutable sum = 0
                        for lane = 0 to 15 do
                            let index = group * 16 + lane
                            let quant = int (MathF.Round(values[index] * inverse, MidpointRounding.ToEven)) |> max -127 |> min 127
                            result.Quants[block * 256 + index] <- int16 quant
                            sum <- sum + quant
                        result.Sums[block * 16 + group] <- sum
            if not valid then result.ValidTokens[block / (columns / 256)] <- false
        result

