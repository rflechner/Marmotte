// First run: dotnet build Marmotte.slnx -p:MarmotteBuildMetal=true (see docs/metal.md).
#r "../src/Marmotte.Metal/bin/Debug/net10.0/Marmotte.Core.dll"
#r "../src/Marmotte.Metal/bin/Debug/net10.0/Marmotte.Metal.dll"

open System
open Marmotte.Core
open Marmotte.Metal

let run () =
    use backend = new MetalBackend() :> ITensorBackend
    let shape = TensorShape.Create [| 2; 3 |]
    use left = backend.Upload(shape, ReadOnlyMemory<single>([| 1.0f; 2.0f; 3.0f; 4.0f; 5.0f; 6.0f |]))
    use right = backend.Upload(shape, ReadOnlyMemory<single>([| 10.0f; 20.0f; 30.0f; 40.0f; 50.0f; 60.0f |]))
    use destination = backend.Allocate(shape)

    backend.Add(left, right, destination)
    backend.Add(destination, left, destination)

    // Download waits for both additions, which may still be running on the GPU at this point.
    let result = Array.zeroCreate<single> shape.ElementCount
    backend.Download(destination, result.AsMemory())
    printfn "Metal Add, then in-place Add: %A" result

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
    printfn "Metal Half Add, then in-place Add: %A" halfResult

    // An F16 tensor added into an F32 one is always requested by name; Add rejects mixed encodings.
    backend.AddF32F16(destination, halfDestination, destination)
    backend.Download(destination, result.AsMemory())
    printfn "Metal AddF32F16 in place: %A" result

run ()
