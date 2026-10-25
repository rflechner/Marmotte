#load "../src/Marmotte.Core/Tensors.fs"
#load "../src/Marmotte.Core/RowDecoder.fs"
#load "../src/Marmotte.Core/WeightMatrix.fs"
#load "../src/Marmotte.Core/RotaryEmbedding.fs"
#load "../src/Marmotte.Core/AttentionTemperature.fs"
#load "../src/Marmotte.Core/Compute.fs"
#load "../src/Marmotte.Cpu/CpuKernels.fs"
#load "../src/Marmotte.Cpu/CpuBackend.fs"
#load "../src/Marmotte.Cuda/CudaBackend.fs"
#load "../src/Marmotte.Metal/MetalBackend.fs"

open System
open Marmotte.Core
open Marmotte.Cpu
open Marmotte.Cuda
open Marmotte.Metal

// The same orchestration uses opaque tensors on every backend. Only the factory changes.
let runAdd (backend: ITensorBackend) =
    let shape = TensorShape.Create [| 2; 3 |]
    use left = backend.Upload(shape, ReadOnlyMemory<single>([| 1.0f; 2.0f; 3.0f; 4.0f; 5.0f; 6.0f |]))
    use right = backend.Upload(shape, ReadOnlyMemory<single>([| 10.0f; 20.0f; 30.0f; 40.0f; 50.0f; 60.0f |]))
    use destination = backend.Allocate(shape)

    backend.Add(left, right, destination)

    let result = Array.zeroCreate<single> shape.ElementCount
    backend.Download(destination, result.AsMemory())
    result

let backends =
    [ "CPU", (fun () -> new CpuBackend() :> ITensorBackend)
      "CUDA", (fun () -> new CudaBackend() :> ITensorBackend)
      "Metal", (fun () -> new MetalBackend() :> ITensorBackend) ]

for name, createBackend in backends do
    use backend = createBackend ()
    try
        printfn "%s Add: %A" name (runAdd backend)
    with :? NotImplementedException as error ->
        // GPU placeholders fail at the first transfer, before a tensor or kernel is created.
        printfn "%s: %s" name error.Message
