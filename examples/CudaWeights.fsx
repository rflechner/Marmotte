// First run: dotnet build Marmotte.slnx -p:MarmotteBuildCuda=true (see docs/cuda.md).
#r "../src/Marmotte.Cuda/bin/Debug/net10.0/Marmotte.Core.dll"
#r "../src/Marmotte.Cuda/bin/Debug/net10.0/Marmotte.Cuda.dll"
#r "../src/Marmotte.Gguf/bin/Debug/net10.0/Marmotte.Gguf.dll"

open System
open System.Diagnostics
open System.IO
open System.Text
open Marmotte.Core
open Marmotte.Cuda
open Marmotte.Gguf

let runExample () =
    let path = Path.Combine(Path.GetTempPath(), $"marmotte-weights-{Guid.NewGuid():N}.bin")
    try
        // Two Q4_0 rows of 32 values: 18 bytes per row, including the two-byte scale.
        File.WriteAllBytes(path, Array.zeroCreate 36)
        let descriptor =
            { Name = "example.weight"
              Encoding = Some TensorEncoding.Q4_0
              SourceEncoding = "Q4_0"
              Shape = [| 2UL; 32UL |]
              ElementCount = 64UL
              ByteSize = Some 36UL }
        use backend = new CudaBackend() :> ITensorBackend
        let upload () =
            use source = new FileTensorSource(File.OpenRead(path), [| { Descriptor = descriptor; AbsoluteOffset = 0L } |]) :> ITensorSource
            backend.UploadWeights(WeightMatrix.Read(source, descriptor.Name))
        use weights = upload ()
        // Upload has finished. The source is closed and the device owns a separate byte copy.
        File.Delete(path)
        printfn "%s: %A, %d x %d, %A" weights.Name weights.Encoding weights.Rows weights.Columns weights.Backend
    finally
        if File.Exists(path) then File.Delete(path)

// Reproduce A.2's payload workload without a model download. Never replace an existing file.
let createBenchmark (path: string) =
    let rows = 155325
    let columns = 3072
    let matrixBytes = int64 rows * int64 (columns / 256) * 144L
    let vectorBytes = columns * 4
    use file = new FileStream(path, FileMode.CreateNew, FileAccess.Write, FileShare.None)
    use writer = new BinaryWriter(file, Encoding.UTF8, true)
    writer.Write(0x46554747u)
    writer.Write(3u)
    writer.Write(9UL)
    writer.Write(0UL)
    let writeName (value: string) =
        let bytes = Encoding.UTF8.GetBytes(value)
        writer.Write(uint64 bytes.Length)
        writer.Write(bytes)
    for i in 0..7 do
        writeName $"matrix.{i}.weight"
        writer.Write(2u)
        writer.Write(uint64 columns)
        writer.Write(uint64 rows)
        writer.Write(12u) // Q4_K; see the stored layout references in docs/gguf.md.
        writer.Write(uint64 (int64 i * matrixBytes))
    writeName "norm.weight"
    writer.Write(1u)
    writer.Write(uint64 columns)
    writer.Write(0u)
    writer.Write(uint64 (8L * matrixBytes))
    while file.Position % 32L <> 0L do writer.Write(0uy)
    let dataOffset = file.Position
    let block = Array.init (1024 * 1024) (fun i -> byte (i % 251))
    let mutable remaining = 8L * matrixBytes + int64 vectorBytes
    while remaining > 0L do
        let count = int (min remaining (int64 block.Length))
        file.Write(block, 0, count)
        remaining <- remaining - int64 count
    printfn "Created %s (%d bytes). Native probe arguments: %d %d %d" path file.Length dataOffset matrixBytes vectorBytes

let measure (path: string) =
    use backend = new CudaBackend() :> ITensorBackend
    use warmup = backend.Upload(TensorShape.Create [| 1 |], ReadOnlyMemory<single>([| 0.0f |]))
    for trial in 1..5 do
        GC.Collect()
        let allocated = GC.GetAllocatedBytesForCurrentThread()
        let watch = Stopwatch.StartNew()
        use source = GgufTensorSource.openMappedFile path
        let openMs = watch.Elapsed.TotalMilliseconds
        let weights = ResizeArray<IWeightTensor>()
        try
            for tensor in source.Tensors do
                if tensor.Shape.Length = 2 then
                    weights.Add(backend.UploadWeights(WeightMatrix.Read(source, tensor.Name)))
                else
                    // A.2 includes one normalization vector. Touch it on the host as loading does.
                    let payload = source.ReadPayload(tensor.Name)
                    if payload.Length > 0 then payload.Span[0] |> ignore
            backend.Synchronize()
            printfn "Trial %d: open %.2f ms, upload %.2f ms, managed allocations %.2f MiB"
                trial openMs (watch.Elapsed.TotalMilliseconds - openMs)
                (float (GC.GetAllocatedBytesForCurrentThread() - allocated) / 1048576.0)
        finally
            for weight in weights do weight.Dispose()

match fsi.CommandLineArgs |> Array.skip 1 with
| [||] -> runExample ()
| [| "--create-benchmark"; path |] -> createBenchmark path
| [| "--measure"; path |] -> measure path
| _ -> failwith "Usage: CudaWeights.fsx [--create-benchmark <new.gguf> | --measure <model.gguf>]"
