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
#load "../src/Marmotte.Cpu/CpuBackend.fs"

open System
open Marmotte.Core
open Marmotte.Cpu

// Key-value cache encodings, chosen separately for keys and values. The first table sizes the
// target model's cache from its layout alone, without allocating it. The second stores random
// keys in a small CPU cache and reports the errors after reading them back.
let encodings =
    [ TensorEncoding.F32
      TensorEncoding.F16
      TensorEncoding.BF16
      TensorEncoding.Q8_0
      TensorEncoding.Q5_1
      TensorEncoding.Q5_0
      TensorEncoding.Q4_1
      TensorEncoding.IQ4_NL
      TensorEncoding.Q4_0 ]

/// Ministral 3 3B: 26 layers with 8 key-value heads of 128 values.
let ministral encoding capacity =
    KeyValueCacheLayout.Create(
        layers = 26,
        capacity = capacity,
        keyValueHeads = 8,
        headDimension = 128,
        keyEncoding = encoding,
        valueEncoding = encoding)

let printSizes () =
    let mebibytes (bytes: int64) = float bytes / 1048576.0
    printfn "Ministral 3 3B cache, keys and values in the same encoding:"
    printfn "%-6s %14s %16s %17s" "" "per token" "4,096 tokens" "32,768 tokens"
    for encoding in encodings do
        let layout = ministral encoding 4096
        let longLayout = ministral encoding 32768
        printfn "%-6s %10.2f KiB %12.0f MiB %13.0f MiB"
            (string encoding) (float layout.BytesPerPosition / 1024.0) (mebibytes layout.TotalBytes) (mebibytes longLayout.TotalBytes)

/// Stores 16 positions of normally distributed keys (mean 0, standard deviation 1), reads them
/// back and returns the root mean square and largest errors. Real keys also have outlier
/// channels, so this only compares the precision of the encodings.
let decodingErrors encoding =
    use backend = new CpuBackend() :> ITensorBackend
    let layout = KeyValueCacheLayout.Create(1, 16, 8, 128, encoding, encoding)
    use cache = backend.AllocateKeyValueCache(layout)
    let random = Random(2512)
    let normal () = sqrt (-2.0 * log (1.0 - random.NextDouble())) * cos (Math.Tau * random.NextDouble())
    let original = Array.init (16 * layout.RowValues) (fun _ -> float32 (normal ()))
    let shape = TensorShape.Create [| 16; layout.RowValues |]
    use keys = backend.Upload(shape, ReadOnlyMemory<single>(original))
    use readKeys = backend.Allocate(shape)
    use readValues = backend.Allocate(shape)

    backend.StoreKeyValues(cache, 0, 0, keys, keys)
    backend.ReadKeyValues(cache, 0, 0, readKeys, readValues)

    let decoded = Array.zeroCreate<single> original.Length
    backend.Download(readKeys, decoded.AsMemory())
    let errors = Array.map2 (fun (expected: single) (actual: single) -> abs (float expected - float actual)) original decoded
    sqrt (errors |> Array.averageBy (fun error -> error * error)), Array.max errors

printSizes ()
printfn ""
printfn "Errors for normally distributed keys:"
printfn "%-6s %10s %10s" "" "RMS" "largest"
for encoding in encodings do
    let rootMeanSquare, largest = decodingErrors encoding
    printfn "%-6s %10.6f %10.6f" (string encoding) rootMeanSquare largest
