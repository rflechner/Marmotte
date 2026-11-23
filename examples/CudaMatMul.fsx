// Build first: dotnet build Marmotte.slnx -p:MarmotteBuildCuda=true
// Then run: dotnet fsi examples/CudaMatMul.fsx
#r "../src/Marmotte.Cuda/bin/Debug/net10.0/Marmotte.Core.dll"
#r "../src/Marmotte.Cuda/bin/Debug/net10.0/Marmotte.Cuda.dll"
#r "../src/Marmotte.Cpu/bin/Debug/net10.0/Marmotte.Cpu.dll"

open System
open System.Collections.Generic
open Marmotte.Core
open Marmotte.Cpu
open Marmotte.Cuda

// A tiny stored F32 matrix: 3 rows of 4 values, little-endian.
let rows = 3
let columns = 4
let weightValues = [| 1.0f; -2.0f; 0.5f; 3.0f
                      0.0f; 1.0f; -1.0f; 2.0f
                      -0.5f; 4.0f; 1.5f; -2.0f |]
let payload = weightValues |> Array.collect BitConverter.GetBytes

let descriptor =
    { Name = "blk.0.attn_q.weight"
      Encoding = Some TensorEncoding.F32
      SourceEncoding = "F32"
      Shape = [| uint64 rows; uint64 columns |]
      ElementCount = uint64 rows * uint64 columns
      ByteSize = Some (uint64 payload.Length) }

let source =
    { new ITensorSource with
        member _.Tensors = [| descriptor |] :> IReadOnlyList<TensorDescriptor>
        member _.TryFindTensor(name) = if name = descriptor.Name then Some descriptor else None
        member _.ReadPayload(_) = ReadOnlyMemory<byte>(payload)
        member _.Dispose() = () }

let matrix = WeightMatrix.Read(source, descriptor.Name)

// Two prompt tokens; the first one doubles as the single-token generation step.
let batch = [| 1.0f; 0.5f; -1.0f; 2.0f
               0.0f; 1.0f; 1.0f; -0.25f |]
let single = batch[.. columns - 1]

// Runs MatVec on one token and MatMul on the two-token batch with the same weights.
let runOn (backend: ITensorBackend) =
    use weights = backend.UploadWeights(matrix)
    let matVecOf (input: single array) =
        use inputTensor = backend.Upload(TensorShape.Create [| columns |], ReadOnlyMemory<single>(input))
        use destination = backend.Allocate(TensorShape.Create [| rows |])
        backend.MatVec(weights, inputTensor, destination)
        let values = Array.zeroCreate<single> rows
        backend.Download(destination, values.AsMemory())
        values
    let matMulOf tokens (input: single array) =
        use inputTensor = backend.Upload(TensorShape.Create [| tokens; columns |], ReadOnlyMemory<single>(input))
        use destination = backend.Allocate(TensorShape.Create [| tokens; rows |])
        backend.MatMul(weights, inputTensor, destination)
        let values = Array.zeroCreate<single> (tokens * rows)
        backend.Download(destination, values.AsMemory())
        values
    matVecOf single, matMulOf 2 batch

let compare () =
    use cpu = new CpuBackend(CpuMatrixProductMode.DecodedFloat) :> ITensorBackend
    use cuda = new CudaBackend() :> ITensorBackend
    let cpuVec, cpuMul = runOn cpu
    let cudaVec, cudaMul = runOn cuda

    printfn "MatVec CPU:  [%s]" (cpuVec |> Array.map string |> String.concat "; ")
    printfn "MatVec CUDA: [%s]" (cudaVec |> Array.map string |> String.concat "; ")
    printfn "MatMul CPU:  [%s]" (cpuMul |> Array.map string |> String.concat "; ")
    printfn "MatMul CUDA: [%s]" (cudaMul |> Array.map string |> String.concat "; ")

    let maxError =
        Array.map2 (fun (a: single) (b: single) -> abs (float a - float b)) cpuMul cudaMul
        |> Array.append (Array.map2 (fun (a: single) (b: single) -> abs (float a - float b)) cpuVec cudaVec)
        |> Array.max
    printfn "Maximum absolute difference from CPU = %g" maxError

compare ()
