// Build first: dotnet build Marmotte.slnx -p:MarmotteBuildCuda=true
// Then run: dotnet fsi examples/CudaRope.fsx
#r "../src/Marmotte.Cuda/bin/Debug/net10.0/Marmotte.Core.dll"
#r "../src/Marmotte.Cuda/bin/Debug/net10.0/Marmotte.Cuda.dll"
#r "../src/Marmotte.Cpu/bin/Debug/net10.0/Marmotte.Cpu.dll"

open System
open Marmotte.Core
open Marmotte.Cpu
open Marmotte.Cuda

let shape = TensorShape.Create [| 3; 8; 128 |]
let positions = [| 0; 16384; 262143 |]
let values = Array.init shape.ElementCount (fun i -> single (i % 73 - 36) / 41.0f)

let check label embedding =
    use cpu = new CpuBackend() :> ITensorBackend
    use cuda = new CudaBackend() :> ITensorBackend
    use cpuTensor = cpu.Upload(shape, ReadOnlyMemory<single>(values))
    use cudaTensor = cuda.Upload(shape, ReadOnlyMemory<single>(values))

    // Same Core contract and positions on both backends. In-place rotation is safe.
    cpu.Rope(cpuTensor, ReadOnlyMemory<int>(positions), embedding, cpuTensor)
    cuda.Rope(cudaTensor, ReadOnlyMemory<int>(positions), embedding, cudaTensor)

    let read (backend: ITensorBackend) (tensor: ITensor) =
        let output = Array.zeroCreate<single> shape.ElementCount
        backend.Download(tensor, output.AsMemory())
        output

    let expected = read cpu cpuTensor
    let actual = read cuda cudaTensor
    let maxError =
        Array.map2 (fun (a: single) (b: single) -> abs (float a - float b)) expected actual
        |> Array.max
    printfn "%s: maximum absolute difference from CPU = %.9g" label maxError

let standard = RotaryEmbedding.Standard(128, 1000000.0, RopePairing.AdjacentPairs)
check "Plain RoPE" standard

let yarn =
    RotaryEmbedding.Yarn(
        128, 1000000.0, RopePairing.AdjacentPairs,
        { ScalingFactor = 16.0
          OriginalContextLength = 16384
          BetaFast = 32.0
          BetaSlow = 1.0
          LogMultiplier = 1.0
          AttentionFactor = 1.0 })
check "YaRN RoPE" yarn
