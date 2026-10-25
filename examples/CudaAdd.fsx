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

    let leftHalf = [| 1.0f; 2.0f; 3.0f; 4.0f; 5.0f; 6.0f |] |> Array.map Half.op_Explicit
    let rightHalf = [| 10.0f; 20.0f; 30.0f; 40.0f; 50.0f; 60.0f |] |> Array.map Half.op_Explicit
    use halfLeft = backend.Upload(shape, ReadOnlyMemory<Half>(leftHalf))
    use halfRight = backend.Upload(shape, ReadOnlyMemory<Half>(rightHalf))
    use halfDestination = backend.Allocate(shape, TensorEncoding.F16)
    backend.Add(halfLeft, halfRight, halfDestination)
    backend.Add(halfDestination, halfLeft, halfDestination)

    // F16 tensors and their transfer buffers use two bytes per element, with no F32 shadow copy.
    let halfResult = Array.zeroCreate<Half> shape.ElementCount
    backend.Download(halfDestination, halfResult.AsMemory())
    printfn "CUDA Half Add, then in-place Add: %A" halfResult

run ()
