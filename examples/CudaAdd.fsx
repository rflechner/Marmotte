// First run: dotnet build Marmotte.slnx -p:MarmotteBuildCuda=true (see docs/cuda.md).
#r "../src/Marmotte.Cuda/bin/Debug/net10.0/Marmotte.Core.dll"
#r "../src/Marmotte.Cuda/bin/Debug/net10.0/Marmotte.Cuda.dll"

open System
open Marmotte.Core
open Marmotte.Cuda

let run () =
    use backend = new CudaBackend(deviceIndex = 0) :> ITensorBackend
    let shape = TensorShape.Create [| 2; 3 |]
    use left = backend.Upload(shape, ReadOnlyMemory<single>([| 1.0f; 2.0f; 3.0f; 4.0f; 5.0f; 6.0f |]))
    use right = backend.Upload(shape, ReadOnlyMemory<single>([| 10.0f; 20.0f; 30.0f; 40.0f; 50.0f; 60.0f |]))
    use destination = backend.Allocate(shape)

    backend.Add(left, right, destination)
    backend.Add(destination, left, destination)

    // Download waits for both additions; intermediate results stay in device memory.
    let result = Array.zeroCreate<single> shape.ElementCount
    backend.Download(destination, result.AsMemory())
    printfn "CUDA Add, then in-place Add: %A" result

run ()
