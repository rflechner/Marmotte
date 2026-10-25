module Marmotte.Backends.YarnTests

open System
open Marmotte.Core
open Marmotte.Cpu
open Xunit

let private target =
    { ScalingFactor = 16.0
      OriginalContextLength = 16384
      BetaFast = 32.0
      BetaSlow = 1.0
      LogMultiplier = 1.0
      AttentionFactor = 1.0 }

let private rotate (dimensions: int array) (positions: int array) (values: single array) pairing parameters =
    use backend = new CpuBackend() :> ITensorBackend
    use tensor = backend.Upload(TensorShape.Create dimensions, ReadOnlyMemory<single>(values))
    let embedding = RotaryEmbedding.Yarn(128, 1000000.0, pairing, parameters)
    backend.Rope(tensor, ReadOnlyMemory<int>(positions), embedding, tensor)
    let output = Array.zeroCreate<single> values.Length
    backend.Download(tensor, output.AsMemory())
    output

[<Theory>]
[<InlineData(false)>]
[<InlineData(true)>]
let ``YaRN rotates query and key heads at original and extended context positions`` splitHalves =
    let pairing = if splitHalves then RopePairing.SplitHalves else RopePairing.AdjacentPairs
    let positions = [| 0; 1; 16383; 16384; 262143 |]
    let random = Random(3201)
    for heads in [ 32; 8 ] do
        let values = Array.init (positions.Length * heads * 128) (fun _ -> single (2.0 * random.NextDouble() - 1.0))
        let output = rotate [| positions.Length; heads; 128 |] positions values pairing target
        for token in 0 .. positions.Length - 1 do
            for head in 0 .. heads - 1 do
                for pair in 0 .. 63 do
                    // Independently specified transition: unchanged through 20, divided by 16 from 37.
                    let multiplier =
                        if pair <= 20 then 1.0
                        elif pair >= 37 then 1.0 / 16.0
                        else 1.0 - 15.0 * float (pair - 20) / 272.0
                    let angle = float positions[token] * Math.Exp(-float pair / 64.0 * Math.Log 1000000.0) * multiplier
                    let start = (token * heads + head) * 128
                    let first, second =
                        if splitHalves then start + pair, start + pair + 64
                        else start + 2 * pair, start + 2 * pair + 1
                    let a, b = float values[first], float values[second]
                    let expectedFirst = a * cos angle - b * sin angle
                    let expectedSecond = a * sin angle + b * cos angle
                    let tolerance = 1e-6 * (abs a + abs b) + 1e-30
                    Assert.InRange(abs (float output[first] - expectedFirst), 0.0, tolerance)
                    Assert.InRange(abs (float output[second] - expectedSecond), 0.0, tolerance)

[<Theory>]
[<InlineData(0.0, 1.2772588722239782)>]
[<InlineData(0.5, 1.121751143713058)>]
[<InlineData(1.0, 1.0)>]
[<InlineData(2.0, 0.8216431602571131)>]
let ``YaRN applies its magnitude to both Q and K so their score scales quadratically`` logMultiplier expectedMagnitude =
    let parameters = { target with LogMultiplier = logMultiplier; AttentionFactor = 1.5 }
    let query = Array.init 128 (fun index -> single (index % 7 - 3))
    let key = Array.init 128 (fun index -> single (index % 3 - 1))
    let baseline = rotate [| 2; 128 |] [| 16384; 16380 |] (Array.append query key) RopePairing.AdjacentPairs target
    let scaled = rotate [| 2; 128 |] [| 16384; 16380 |] (Array.append query key) RopePairing.AdjacentPairs parameters
    let product (values: single array) =
        Array.init 128 (fun index -> float values[index] * float values[128 + index]) |> Array.sum
    let magnitude = 1.5 * expectedMagnitude
    Assert.InRange(abs (product scaled - product baseline * magnitude * magnitude), 0.0, 5e-5)

// Captured by running the unmodified ggml CPU kernel, llama.cpp commit
// 2280f0fd163b03d4331d49e08460f66502e35854 (2026-10-25), on an F32 [128; 1; 4] tensor
// whose every adjacent pair is (1, 0). ggml_rope_ext: NORMAL, n_dims=128, n_ctx_orig=16384,
// freq_base=1e6, freq_scale=1/16, ext_factor=1, attn_factor=1/(1+0.1*ln(16)), betas=32/1.
// Values below are output data, not values computed by Marmotte or a port of ggml's kernel.
[<Theory>]
[<InlineData(0, 0, 1.0, 0.0)>]
[<InlineData(0, 27, 1.0, 0.0)>]
[<InlineData(1, 0, 0.540302277, 0.841470957)>]
[<InlineData(1, 20, 0.999911070, 0.0133348117)>]
[<InlineData(1, 27, 0.999998391, 0.00180674531)>]
[<InlineData(1, 37, 1.0, 2.12387749e-5)>]
[<InlineData(1, 63, 1.0, 7.75584468e-8)>]
[<InlineData(16384, 0, -0.828534186, -0.559938490)>]
[<InlineData(16384, 20, 0.142844334, -0.989745140)>]
[<InlineData(16384, 27, -0.241003662, -0.970524192)>]
[<InlineData(16384, 37, 0.940064788, 0.340995908)>]
[<InlineData(16384, 63, 0.999999166, 0.00127071724)>]
[<InlineData(262143, 0, -0.609161496, 0.793046176)>]
[<InlineData(262143, 20, -0.650617659, 0.759405494)>]
[<InlineData(262143, 27, -0.728520811, 0.685023665)>]
[<InlineData(262143, 37, 0.754706740, -0.656062305)>]
[<InlineData(262143, 63, 0.999793351, 0.0203300007)>]
let ``target YaRN agrees with recorded ggml rotations within F32 angle rounding`` position pair expectedFirst expectedSecond =
    let values = Array.init 128 (fun index -> if index % 2 = 0 then 1.0f else 0.0f)
    let output = rotate [| 1; 128 |] [| position |] values RopePairing.AdjacentPairs target
    // ggml repeatedly multiplies its angle in F32; Marmotte prepares frequencies and angles in
    // double precision. The error bound grows with the number of multiplications and the angle.
    let frequency = RotaryEmbedding.Yarn(128, 1000000.0, RopePairing.AdjacentPairs, target).Frequencies[pair]
    let tolerance = 2e-7 + 2e-7 * float (pair + 2) * float position * frequency
    Assert.InRange(abs (float output[2 * pair] - expectedFirst), 0.0, tolerance)
    Assert.InRange(abs (float output[2 * pair + 1] - expectedSecond), 0.0, tolerance)

    if position = 1 then
        // Recover the frequency from a single turn to check even the smallest frequencies with
        // relative precision; an absolute F32 tolerance alone would hide a wrong slow-band scale.
        let referenceFrequency = Math.Atan2(expectedSecond, expectedFirst)
        Assert.InRange(abs (frequency / referenceFrequency - 1.0), 0.0, 5e-6)
