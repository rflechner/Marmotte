#load "../src/Marmotte.Core/Tensors.fs"
#load "../src/Marmotte.Core/RowDecoder.fs"
#load "../src/Marmotte.Core/RowEncoder.fs"
#load "../src/Marmotte.Core/WeightMatrix.fs"
#load "../src/Marmotte.Core/RotaryEmbedding.fs"
#load "../src/Marmotte.Core/AttentionTemperature.fs"
#load "../src/Marmotte.Core/KeyValueCache.fs"
#load "../src/Marmotte.Core/Compute.fs"
#load "../src/Marmotte.Core/TensorBackend.fs"
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

// Half buffers stay at two bytes per element. Converting these small literals is explicit.
let runHalfAdd (backend: ITensorBackend) =
    let shape = TensorShape.Create [| 2; 3 |]
    let leftValues = [| 1.0f; 2.0f; 3.0f; 4.0f; 5.0f; 6.0f |] |> Array.map Half.op_Explicit
    let rightValues = [| 10.0f; 20.0f; 30.0f; 40.0f; 50.0f; 60.0f |] |> Array.map Half.op_Explicit
    use left = backend.Upload(shape, ReadOnlyMemory<Half>(leftValues))
    use right = backend.Upload(shape, ReadOnlyMemory<Half>(rightValues))
    use destination = backend.Allocate(shape, TensorEncoding.F16)

    backend.Add(left, right, destination)

    let result = Array.zeroCreate<Half> shape.ElementCount
    backend.Download(destination, result.AsMemory())
    result

// GPU examples load their compiled assemblies and native libraries:
// examples/CudaAdd.fsx and examples/MetalAdd.fsx.
let backends = [ ("CPU", (fun () -> new CpuBackend() :> ITensorBackend)) ]

for name, createBackend in backends do
    use backend = createBackend ()
    printfn "%s Add: %A" name (runAdd backend)
    printfn "%s Half Add: %A" name (runHalfAdd backend)
