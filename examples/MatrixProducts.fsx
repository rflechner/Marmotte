#r "../src/Marmotte.Core/bin/Release/net10.0/Marmotte.Core.dll"
#r "../src/Marmotte.Cpu/bin/Release/net10.0/Marmotte.Cpu.dll"

open System
open System.Buffers.Binary
open System.Collections.Generic
open System.Diagnostics
open System.Runtime.InteropServices
open System.Runtime.Intrinsics
open Marmotte.Core
open Marmotte.Cpu

// Build Release first. Timings exclude fixture construction, upload and download.
// Synthetic finite blocks exercise decoding without requiring model weights.
let matrix (encoding: TensorEncoding) rows columns =
    let random = Random(42)
    let bytes = Array.zeroCreate<byte> (rows * columns / encoding.BlockElements * encoding.BlockBytes)
    match encoding with
    | TensorEncoding.F32 ->
        for index in 0 .. rows * columns - 1 do
            BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(index * 4, 4), float32 (random.NextDouble() - 0.5))
    | TensorEncoding.Q4_K | TensorEncoding.Q6_K ->
        random.NextBytes bytes
        for offset in 0 .. encoding.BlockBytes .. bytes.Length - 1 do
            let scaleOffsets = if encoding = TensorEncoding.Q4_K then [ 0; 2 ] else [ 208 ]
            for scaleOffset in scaleOffsets do
                BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(offset + scaleOffset, 2), 0x2400us)
    | _ -> invalidArg "encoding" "This benchmark uses F32, Q4_K and Q6_K."
    let descriptor =
        { Name = "weights"
          Encoding = Some encoding
          SourceEncoding = string encoding
          Shape = [| uint64 rows; uint64 columns |]
          ElementCount = uint64 rows * uint64 columns
          ByteSize = Some (uint64 bytes.Length) }
    use source =
        { new ITensorSource with
            member _.Tensors = [| descriptor |] :> IReadOnlyList<_>
            member _.TryFindTensor name = if name = descriptor.Name then Some descriptor else None
            member _.ReadPayload _ = ReadOnlyMemory<byte>(bytes)
            member _.Dispose() = () }
    WeightMatrix.Read(source, descriptor.Name)

let measure encoding rows columns tokens =
    use backend = new CpuBackend() :> ITensorBackend
    use weights = backend.UploadWeights(matrix encoding rows columns)
    let values = Array.init (tokens * columns) (fun i -> float32 (i % 17 - 8) / 16.0f)
    use input = backend.Upload(TensorShape.Create [| tokens; columns |], ReadOnlyMemory<single>(values))
    use output = backend.Allocate(TensorShape.Create [| tokens; rows |])
    let run () = backend.MatMul(weights, input, output)
    for _ in 1 .. 10 do run ()
    let samples =
        Array.init 7 (fun _ ->
            let timer = Stopwatch.StartNew()
            for _ in 1 .. 10 do run ()
            timer.Elapsed.TotalMilliseconds / 10.0)
        |> Array.sort
    let result = Array.zeroCreate<single> (rows * tokens)
    backend.Download(output, Memory<single>(result))
    printfn "%A %dx%d batch=%d: median %.3f ms, range %.3f..%.3f; checksum %.6g"
        encoding rows columns tokens samples[3] samples[0] samples[6] (Array.sumBy float result)

printfn "%s %A; processors=%d; Vector128=%b Vector256=%b"
    RuntimeInformation.FrameworkDescription RuntimeInformation.ProcessArchitecture Environment.ProcessorCount
    Vector128.IsHardwareAccelerated Vector256.IsHardwareAccelerated
for rows, columns in [ 16, 64; 64, 256; 1024, 3072 ] do
    for encoding in [ TensorEncoding.F32; TensorEncoding.Q4_K; TensorEncoding.Q6_K ] do
        if columns % encoding.BlockElements = 0 then
            for tokens in [ 1; 8 ] do measure encoding rows columns tokens
