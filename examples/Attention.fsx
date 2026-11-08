#load "../src/Marmotte.Core/Tensors.fs"
#load "../src/Marmotte.Core/RowDecoder.fs"
#load "../src/Marmotte.Core/RowEncoder.fs"
#load "../src/Marmotte.Core/WeightMatrix.fs"
#load "../src/Marmotte.Core/RotaryEmbedding.fs"
#load "../src/Marmotte.Core/AttentionTemperature.fs"
#load "../src/Marmotte.Core/KeyValueCache.fs"
#load "../src/Marmotte.Core/Compute.fs"
#load "../src/Marmotte.Cpu/CpuKernels.fs"
#load "../src/Marmotte.Cpu/CpuStorage.fs"
#load "../src/Marmotte.Cpu/CpuTensor.fs"
#load "../src/Marmotte.Cpu/CpuWeightTensor.fs"
#load "../src/Marmotte.Cpu/CpuKeyValueCache.fs"
#load "../src/Marmotte.Cpu/CpuQ8Activations.fs"
#load "../src/Marmotte.Cpu/CpuKQuantBlocks.fs"
#load "../src/Marmotte.Cpu/CpuMatrixKernels.fs"
#load "../src/Marmotte.Cpu/CpuBackend.fs"

open System
open Marmotte.Core
open Marmotte.Cpu

// Four query heads share two KV heads. Zero queries give uniform causal weights,
// making the output the average of visible values for each head's group.
let run () =
    use backend = new CpuBackend() :> ITensorBackend
    let layout = KeyValueCacheLayout.Create(1, 3, 2, 2, TensorEncoding.F32, TensorEncoding.F32)
    use cache = backend.AllocateKeyValueCache(layout)
    let rows = TensorShape.Create [| 3; 2; 2 |]
    use keys = backend.Allocate(rows)
    use values = backend.Upload(rows, ReadOnlyMemory<single>([|
        2.0f; 4.0f; 20.0f; 40.0f
        6.0f; 8.0f; 60.0f; 80.0f
        10.0f; 12.0f; 100.0f; 120.0f
    |]))
    backend.StoreKeyValues(cache, 0, 0, keys, values)
    use queries = backend.Allocate(TensorShape.Create [| 3; 4; 2 |])
    use output = backend.Allocate(queries.Shape)
    backend.Attention(cache, 0, 0, queries, output)
    let result = Array.zeroCreate<single> output.Shape.ElementCount
    backend.Download(output, result.AsMemory())
    for token = 0 to 2 do
        printfn "Position %d: %A" token result[token * 8 .. token * 8 + 7]

run ()
