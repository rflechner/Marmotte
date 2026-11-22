// Build first: dotnet build Marmotte.slnx -p:MarmotteBuildCuda=true
// Then run: dotnet fsi examples/CudaEmbedding.fsx
#r "../src/Marmotte.Cuda/bin/Debug/net10.0/Marmotte.Core.dll"

#r "../src/Marmotte.Cuda/bin/Debug/net10.0/Marmotte.Cuda.dll"

#r "../src/Marmotte.Cpu/bin/Debug/net10.0/Marmotte.Cpu.dll"

open System
open System.Collections.Generic
open Marmotte.Core
open Marmotte.Cpu
open Marmotte.Cuda

// A tiny stored Q8_0 vocabulary: 4 rows of 32 values, one 34-byte block per row.
// Each block is an F16 scale followed by 32 signed codes; F16 scales are little-endian.
let block scaleByte codes =
    let bytes = Array.zeroCreate<byte> 34
    bytes[0] <- 0x00uy
    bytes[1] <- scaleByte // 0x34 = 0.25, 0x38 = 0.5, 0x3C = 1.0, 0xC0 = -2.0
    codes |> Array.iteri (fun index code -> bytes[2 + index] <- byte code)
    bytes

let payload =
    [| block 0x34uy [| for i in 0 .. 31 -> sbyte (i - 16) |]   // row 0, scale 0.25
       block 0x38uy [| for i in 0 .. 31 -> sbyte (16 - i) |]   // row 1, scale 0.5
       block 0xC0uy [| for i in 0 .. 31 -> sbyte (i % 7 - 3) |] // row 2, scale -2
       block 0x3Cuy [| for i in 0 .. 31 -> sbyte 1 |] |]        // row 3, scale 1
    |> Array.concat

let descriptor =
    { Name = "token_embd.weight"
      Encoding = Some TensorEncoding.Q8_0
      SourceEncoding = "Q8_0"
      Shape = [| 4UL; 32UL |]
      ElementCount = 128UL
      ByteSize = Some (uint64 payload.Length) }

let source =
    { new ITensorSource with
        member _.Tensors = [| descriptor |] :> IReadOnlyList<TensorDescriptor>
        member _.TryFindTensor(name) = if name = descriptor.Name then Some descriptor else None
        member _.ReadPayload(_) = ReadOnlyMemory<byte>(payload)
        member _.Dispose() = () }

let matrix = WeightMatrix.Read(source, descriptor.Name)
let tokenIds = [| 2; 0; 2; 3 |]

let lookup (backend: ITensorBackend) =
    use weights = backend.UploadWeights(matrix)
    use destination = backend.Allocate(TensorShape.Create [| tokenIds.Length; matrix.Columns |])
    backend.EmbeddingLookup(weights, ReadOnlyMemory<int>(tokenIds), destination)
    let values = Array.zeroCreate<single> destination.Shape.ElementCount
    backend.Download(destination, values.AsMemory())
    values

let run () =
    use cpu = new CpuBackend() :> ITensorBackend
    use cuda = new CudaBackend() :> ITensorBackend

    let expected = lookup cpu
    let actual = lookup cuda

    printfn "Token IDs: %A" tokenIds
    for token in 0 .. tokenIds.Length - 1 do
        let row = actual[token * 32 .. (token + 1) * 32 - 1]
        printfn "Position %d -> row %d: [%s; ...]" token tokenIds[token] (row[..3] |> Array.map string |> String.concat "; ")

    let maxError =
        Array.map2 (fun (a: single) (b: single) -> abs (float a - float b)) expected actual
        |> Array.max
    printfn "Maximum absolute difference from CPU = %g" maxError

run ()
