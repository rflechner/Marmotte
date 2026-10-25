#load "../src/Marmotte.Core/Tensors.fs"
#load "../src/Marmotte.Core/RowDecoder.fs"
#load "../src/Marmotte.Core/WeightMatrix.fs"
#load "../src/Marmotte.Core/RotaryEmbedding.fs"
#load "../src/Marmotte.Core/AttentionTemperature.fs"
#load "../src/Marmotte.Core/Compute.fs"
#load "../src/Marmotte.Cpu/CpuKernels.fs"
#load "../src/Marmotte.Cpu/CpuBackend.fs"

open System
open System.Collections.Generic
open Marmotte.Core
open Marmotte.Cpu

// A tiny stored F32 vocabulary: token 0 = [1; 2; 3], token 1 = [4; 5; 6].
// A real model uses GgufTensorSource or SafetensorsTensorSource through the same contract.
let descriptor =
    { Name = "token_embd.weight"
      Encoding = Some TensorEncoding.F32
      SourceEncoding = "F32"
      Shape = [| 2UL; 3UL |]
      ElementCount = 6UL
      ByteSize = Some 24UL }

let payload = Convert.FromHexString("0000803F0000004000004040000080400000A0400000C040")

let run () =
    use source =
        { new ITensorSource with
            member _.Tensors = [| descriptor |] :> IReadOnlyList<TensorDescriptor>
            member _.TryFindTensor(name) = if name = descriptor.Name then Some descriptor else None
            member _.ReadPayload(_) = ReadOnlyMemory<byte>(payload)
            member _.Dispose() = () }

    use backend = new CpuBackend() :> ITensorBackend
    use weights = backend.UploadWeights(WeightMatrix.Read(source, descriptor.Name))
    let tokenIds = [| 1; 0; 1 |]
    use activations = backend.Allocate(TensorShape.Create [| tokenIds.Length; weights.Columns |])

    backend.EmbeddingLookup(weights, ReadOnlyMemory<int>(tokenIds), activations)

    let result = Array.zeroCreate<single> activations.Shape.ElementCount
    backend.Download(activations, result.AsMemory())
    for index = 0 to tokenIds.Length - 1 do
        printfn "Token %d: %A" tokenIds[index] result[index * weights.Columns .. (index + 1) * weights.Columns - 1]

run ()
