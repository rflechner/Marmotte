#load "../src/Marmotte.Core/Tensors.fs"
#load "../src/Marmotte.Core/RowDecoder.fs"
#load "../src/Marmotte.Core/WeightMatrix.fs"
#load "../src/Marmotte.Core/RotaryEmbedding.fs"
#load "../src/Marmotte.Core/AttentionTemperature.fs"
#load "../src/Marmotte.Core/Compute.fs"
#load "../src/Marmotte.Cpu/CpuKernels.fs"
#load "../src/Marmotte.Cpu/CpuBackend.fs"

open System
open Marmotte.Core
open Marmotte.Cpu

// Explicit target-model configuration; no model file is needed for this example.
let parameters =
    { ScalingFactor = 16.0
      OriginalContextLength = 16384
      BetaFast = 32.0
      BetaSlow = 1.0
      LogMultiplier = 1.0
      AttentionFactor = 1.0 }

let embedding = RotaryEmbedding.Yarn(128, 1000000.0, RopePairing.AdjacentPairs, parameters)
let standard = RotaryEmbedding.Standard(128, 1000000.0, RopePairing.AdjacentPairs)

printfn "YaRN magnitude: %g (the target model's logarithmic factors cancel)" embedding.Magnitude
for pair in [ 0; 20; 27; 37; 63 ] do
    printfn "Pair %2d: plain %.9g -> YaRN %.9g radians/position" pair standard.Frequencies[pair] embedding.Frequencies[pair]

let run () =
    use backend = new CpuBackend() :> ITensorBackend
    let positions = [| 0; 16384; 262143 |]
    let shape = TensorShape.Create [| positions.Length; 1; embedding.HeadDimension |]
    // A unit vector (1, 0) per pair makes the rotated values easy to inspect as (cos, sin).
    let values = Array.init shape.ElementCount (fun index -> if index % 2 = 0 then 1.0f else 0.0f)
    use tensor = backend.Upload(shape, ReadOnlyMemory<single>(values))
    backend.Rope(tensor, ReadOnlyMemory<int>(positions), embedding, tensor)
    backend.Download(tensor, values.AsMemory())

    for token in 0 .. positions.Length - 1 do
        let first = token * embedding.HeadDimension + 2 * 27
        printfn "Position %6d, pair 27: (% .6f, % .6f)" positions[token] values[first] values[first + 1]

// Reuse this embedding when rotating Q and K. V does not receive RoPE.
run ()
