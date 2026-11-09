#r "../src/Marmotte.Core/bin/Release/net10.0/Marmotte.Core.dll"
#r "../src/Marmotte.Cpu/bin/Release/net10.0/Marmotte.Cpu.dll"

open System
open System.Buffers.Binary
open System.Collections.Generic
open System.Diagnostics
open System.Runtime.InteropServices
open Marmotte.Core
open Marmotte.Cpu

// Build Release first. No model download. Timings include kernels/temporary buffers,
// but exclude fixture creation, weight loading, cache allocation and result downloads.
let measure label action =
    for _ in 1 .. 3 do action ()
    let samples =
        Array.init 7 (fun _ ->
            let timer = Stopwatch.StartNew()
            for _ in 1 .. 3 do action ()
            timer.Elapsed.TotalMilliseconds / 3.0)
        |> Array.sort
    printfn "%s: %.3f ms (range %.3f..%.3f)" label samples[3] samples[0] samples[6]

let download (backend: ITensorBackend) (tensor: ITensor) =
    let values = Array.zeroCreate<single> tensor.Shape.ElementCount
    backend.Download(tensor, values.AsMemory())
    values

let attention prefix tokens encoding mode =
    use backend = new CpuBackend(CpuMatrixProductMode.DecodedFloat, mode) :> ITensorBackend
    let dimension, kvHeads, queryHeads = 128, 2, 8
    use cache = backend.AllocateKeyValueCache(KeyValueCacheLayout.Create(1, prefix, kvHeads, dimension, encoding, encoding))
    let upload shape phase count =
        let values = Array.init count (fun i -> single (sin (float i * 0.17 + phase)))
        backend.Upload(TensorShape.Create shape, values.AsMemory())
    use keys = upload [| prefix; kvHeads * dimension |] 0.0 (prefix * kvHeads * dimension)
    use values = upload [| prefix; kvHeads * dimension |] 1.0 (prefix * kvHeads * dimension)
    backend.StoreKeyValues(cache, 0, 0, keys, values)
    use queries = upload [| tokens; queryHeads * dimension |] 2.0 (tokens * queryHeads * dimension)
    use output = backend.Allocate(queries.Shape)
    let run () = backend.Attention(cache, 0, prefix - tokens, queries, output)
    measure $"Attention {mode} {encoding}, prefix={prefix}, tokens={tokens}" run
    // Attention is synchronous and single-threaded, so this captures its allocation.
    let before = GC.GetAllocatedBytesForCurrentThread()
    run ()
    let allocated = GC.GetAllocatedBytesForCurrentThread() - before
    printfn "  Managed allocation: %d bytes/call" allocated
    download backend output

let syntheticModel () =
    let tensors = Dictionary<string, TensorDescriptor * byte array>()
    let weight name shape norm =
        let count = Array.reduce (*) shape
        let bytes = Array.zeroCreate<byte> (count * 4)
        for index = 0 to count - 1 do
            let value = if norm then 1.0f else single (sin (float (index % 997))) * 0.02f
            BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(index * 4, 4), value)
        let descriptor =
            { Name = name; Encoding = Some TensorEncoding.F32; SourceEncoding = "F32"
              Shape = Array.map uint64 shape; ElementCount = uint64 count; ByteSize = Some (uint64 bytes.Length) }
        tensors.Add(name, (descriptor, bytes))
        descriptor
    let embedding = weight "embedding" [| 8192; 128 |] false
    let norm name = weight name [| 128 |] true
    let layers = Array.init 2 (fun index ->
        let prefix = $"layer{index}."
        { AttentionNormalization = norm (prefix + "attentionNorm")
          Query = weight (prefix + "query") [| 128; 128 |] false
          Key = weight (prefix + "key") [| 64; 128 |] false
          Value = weight (prefix + "value") [| 64; 128 |] false
          AttentionOutput = weight (prefix + "attentionOutput") [| 128; 128 |] false
          FeedForwardNormalization = norm (prefix + "feedForwardNorm")
          FeedForwardGate = weight (prefix + "gate") [| 256; 128 |] false
          FeedForwardUp = weight (prefix + "up") [| 256; 128 |] false
          FeedForwardDown = weight (prefix + "down") [| 128; 256 |] false })
    let configuration =
        { Architecture = ModelArchitecture.Mistral3; ContextLength = 64; BlockCount = 2
          EmbeddingLength = 128; FeedForwardLength = 256; AttentionHeadCount = 4
          KeyValueHeadCount = 2; KeyLength = 32; ValueLength = 32
          RmsNormalizationEpsilon = 1e-5f; RopeFrequencyBase = 10000.0
          RopeScaling = ModelRopeScaling.None; AttentionTemperature = None }
    let layout =
        { Configuration = configuration; VocabularySize = 8192; TokenEmbedding = embedding
          OutputNormalization = norm "outputNorm"; Output = embedding; HasTiedOutput = true
          Layers = Array.AsReadOnly layers }
    let source =
        { new ITensorSource with
            member _.Tensors = tensors.Values |> Seq.map fst |> Seq.toArray :> IReadOnlyList<_>
            member _.TryFindTensor name =
                match tensors.TryGetValue name with
                | true, (descriptor, _) -> Some descriptor
                | _ -> None
            member _.ReadPayload name = ReadOnlyMemory<byte>(snd tensors[name])
            member _.Dispose() = () }
    source, layout

printfn "%s %A; processors=%d" RuntimeInformation.FrameworkDescription RuntimeInformation.ProcessArchitecture Environment.ProcessorCount
for encoding in [ TensorEncoding.F32; TensorEncoding.Q8_0 ] do
    for prefix, tokens in [ 128, 8; 2048, 1; 2048, 8 ] do
        let expected = attention prefix tokens encoding CpuAttentionMode.Reference
        let actual = attention prefix tokens encoding CpuAttentionMode.Online
        let error = Array.map2 (fun a b -> abs (a - b)) expected actual |> Array.max
        printfn "  Maximum absolute difference: %.6g" error
        if error > 0.00002f then failwith "Attention differs from the reference."

let measureForward () =
    let source, layout = syntheticModel ()
    use source = source
    use backend = new CpuBackend() :> ITensorBackend
    use model = TransformerModel.Load(backend, source, layout)
    use cache = backend.AllocateKeyValueCache(KeyValueCacheLayout.Create(2, 64, 2, 32, TensorEncoding.F32, TensorEncoding.F32))
    let prompt = ReadOnlyMemory<int>(Array.init 32 id)
    let mutable expectedLast = Array.empty<single>
    for selection in [ LogitSelection.AllTokens; LogitSelection.LastToken; LogitSelection.NoTokens ] do
        let run () =
            use logits = model.Forward(cache, 0, prompt, selection)
            backend.Synchronize()
        measure $"Forward {selection}, 32 tokens, vocabulary=8192" run
        use logits = model.Forward(cache, 0, prompt, selection)
        let result = download backend logits
        printfn "  Logit storage: %d bytes" (result.Length * 4)
        match selection with
        | LogitSelection.AllTokens -> expectedLast <- result[result.Length - layout.VocabularySize ..]
        | LogitSelection.LastToken when result <> expectedLast -> failwith "Selected logits differ."
        | _ -> ()

measureForward ()
