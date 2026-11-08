module Marmotte.Backends.TransformerTests

open System
open System.Buffers.Binary
open System.Collections.Generic
open System.IO
open Marmotte.Core
open Marmotte.Cpu
open Xunit

// Small dense weights with distinct values for every role, encoded without RowEncoder.
type private ModelFixture(tied: bool, halfPrecision: bool, ?quantized: bool) =
    let quantized = defaultArg quantized false
    let embeddingWidth = if quantized then 32 else 4
    let feedForwardWidth = if quantized then 32 else 6
    let queryWidth = if quantized then 32 else 8
    let tensors = Dictionary<string, TensorDescriptor * byte array * double array>()
    let mutable reads = 0
    let mutable disposed = false
    let mutable failName = ""
    let configuration =
        { Architecture = ModelArchitecture.Mistral3; ContextLength = 8; BlockCount = 2
          EmbeddingLength = embeddingWidth; FeedForwardLength = feedForwardWidth; AttentionHeadCount = queryWidth / 2
          KeyValueHeadCount = 2; KeyLength = 2; ValueLength = 2
          RmsNormalizationEpsilon = 1e-5f; RopeFrequencyBase = 10000.0
          RopeScaling = ModelRopeScaling.None; AttentionTemperature = None }
    let weight name shape norm =
        let count = Array.reduce (*) shape
        let seed = tensors.Count + 1
        let values = Array.init count (fun i -> if norm then 0.8f + single i * 0.12f else single (sin (float (i * 7 + seed * 13))) * 0.4f)
        let encoding, bytes =
            if quantized && not norm then TensorEncoding.Q8_0, count / 32 * 34
            elif halfPrecision then TensorEncoding.F16, count * 2
            else TensorEncoding.F32, count * 4
        let payload = Array.zeroCreate<byte> bytes
        let decoded = values |> Array.mapi (fun i value ->
            if encoding = TensorEncoding.Q8_0 then
                let block, within = i / 32, i % 32
                BinaryPrimitives.WriteUInt16LittleEndian(payload.AsSpan(block * 34, 2), 0x2800us)
                let level = sbyte (MathF.Round(value * 32.0f))
                payload[block * 34 + 2 + within] <- byte level
                float level / 32.0
            elif halfPrecision then
                let half = Half.op_Explicit value
                BinaryPrimitives.WriteUInt16LittleEndian(payload.AsSpan(i * 2, 2), BitConverter.HalfToUInt16Bits half)
                float half
            else
                BinaryPrimitives.WriteSingleLittleEndian(payload.AsSpan(i * 4, 4), value)
                float value)
        let descriptor =
            { Name = name; Encoding = Some encoding; SourceEncoding = string encoding
              Shape = Array.map uint64 shape; ElementCount = uint64 count; ByteSize = Some (uint64 payload.Length) }
        tensors.Add(name, (descriptor, payload, decoded))
        descriptor
    let embedding = weight "embedding" [| 5; embeddingWidth |] false
    let output = if tied then embedding else weight "output" [| 5; embeddingWidth |] false
    let outputNorm = weight "outputNorm" [| embeddingWidth |] true
    let layers = Array.init 2 (fun layer ->
        let w name shape norm = weight $"layer{layer}.{name}" shape norm
        { AttentionNormalization = w "attentionNorm" [| embeddingWidth |] true
          Query = w "query" [| queryWidth; embeddingWidth |] false
          Key = w "key" [| 4; embeddingWidth |] false
          Value = w "value" [| 4; embeddingWidth |] false
          AttentionOutput = w "attentionOutput" [| embeddingWidth; queryWidth |] false
          FeedForwardNormalization = w "feedForwardNorm" [| embeddingWidth |] true
          FeedForwardGate = w "gate" [| feedForwardWidth; embeddingWidth |] false
          FeedForwardUp = w "up" [| feedForwardWidth; embeddingWidth |] false
          FeedForwardDown = w "down" [| embeddingWidth; feedForwardWidth |] false })
    member _.Layout =
        { Configuration = configuration; VocabularySize = 5; TokenEmbedding = embedding
          Output = output; OutputNormalization = outputNorm; HasTiedOutput = tied
          Layers = Array.AsReadOnly layers }
    member _.Values(descriptor: TensorDescriptor) = let _, _, values = tensors[descriptor.Name] in values
    member _.Reads = reads
    member _.FailName with set name = failName <- name
    interface ITensorSource with
        member _.Tensors = tensors.Values |> Seq.map (fun (descriptor, _, _) -> descriptor) |> Seq.toArray :> IReadOnlyList<_>
        member _.TryFindTensor name =
            match tensors.TryGetValue name with
            | true, (descriptor, _, _) -> Some descriptor
            | _ -> None
        member _.ReadPayload name =
            ObjectDisposedException.ThrowIf(disposed, name)
            reads <- reads + 1
            if name = failName then raise (IOException("Injected payload failure."))
            let _, payload, _ = tensors[name]
            ReadOnlyMemory<byte>(payload)
    interface IDisposable with
        member _.Dispose() = disposed <- true

let private download (backend: ITensorBackend) (tensor: ITensor) =
    let result = Array.zeroCreate<single> tensor.Shape.ElementCount
    backend.Download(tensor, result.AsMemory())
    result

let private cache (backend: ITensorBackend) encoding =
    backend.AllocateKeyValueCache(KeyValueCacheLayout.Create(2, 8, 2, 2, encoding, encoding))

let private close (expected: double array) (actual: single array) =
    Assert.Equal(expected.Length, actual.Length)
    Array.iter2 (fun expected actual ->
        Assert.True(abs (expected - float actual) < 0.00003, $"Expected {expected}, got {actual}.")) expected actual

// An independent scalar forward calculation. It uses no backend kernels, cache or row codecs.
let private reference (fixture: ModelFixture) (layout: ModelTensorLayout) (ids: int array) =
    let c = layout.Configuration
    let matrix (descriptor: TensorDescriptor) (input: double array) =
        let weights = fixture.Values descriptor
        Array.init (int descriptor.Shape[0]) (fun row ->
            Array.sum (Array.mapi (fun column value -> value * weights[row * input.Length + column]) input))
    let norm descriptor (input: double array) =
        let denominator = sqrt (Array.averageBy (fun value -> value * value) input + float c.RmsNormalizationEpsilon)
        Array.map2 (fun value weight -> value / denominator * weight) input (fixture.Values descriptor)
    let rotate position (input: double array) =
        // One adjacent pair per head. For the fixture's YaRN settings, pair zero is
        // fully extrapolated (frequency 1); the magnitude still changes Q and K.
        let magnitude =
            match c.RopeScaling with
            | ModelRopeScaling.None -> 1.0
            | ModelRopeScaling.Yarn parameters ->
                parameters.AttentionFactor * (1.0 + 0.1 * log parameters.ScalingFactor)
                / (1.0 + 0.1 * parameters.LogMultiplier * log parameters.ScalingFactor)
        Array.init input.Length (fun i ->
            let first = i / 2 * 2
            let a, b = input[first], input[first + 1]
            if i % 2 = 0 then magnitude * (a * cos (float position) - b * sin (float position))
            else magnitude * (a * sin (float position) + b * cos (float position)))
    let embeddings = fixture.Values layout.TokenEmbedding
    let mutable hidden = ids |> Array.map (fun id -> embeddings[id * c.EmbeddingLength .. (id + 1) * c.EmbeddingLength - 1])
    for layer in layout.Layers do
        let normalized = hidden |> Array.map (norm layer.AttentionNormalization)
        let keys = normalized |> Array.mapi (fun p row -> matrix layer.Key row |> rotate p)
        let values = normalized |> Array.map (matrix layer.Value)
        let attention = normalized |> Array.mapi (fun p row ->
            let queries = matrix layer.Query row |> rotate p
            let factor =
                match c.AttentionTemperature with
                | None -> 1.0
                | Some temperature -> 1.0 + temperature.Scale * log (float (p / temperature.FloorLength) + 1.0)
            Array.init (c.AttentionHeadCount * 2) (fun channel ->
                let head, coordinate = channel / 2, channel % 2
                let kv = head / (c.AttentionHeadCount / c.KeyValueHeadCount)
                let scores = Array.init (p + 1) (fun j ->
                    (queries[head * 2] * keys[j][kv * 2] + queries[head * 2 + 1] * keys[j][kv * 2 + 1]) * factor / sqrt 2.0)
                let maximum = Array.max scores
                let weights = scores |> Array.map (fun score -> exp (score - maximum))
                Array.sum (Array.mapi (fun j weight -> weight * values[j][kv * 2 + coordinate]) weights) / Array.sum weights)
            |> matrix layer.AttentionOutput)
        hidden <- Array.map2 (Array.map2 (+)) hidden attention
        hidden <- hidden |> Array.map (fun row ->
            let normalized = norm layer.FeedForwardNormalization row
            let gate = matrix layer.FeedForwardGate normalized
            let up = matrix layer.FeedForwardUp normalized
            let activated = Array.map2 (fun g u -> g / (1.0 + exp -g) * u) gate up
            Array.map2 (+) row (matrix layer.FeedForwardDown activated))
    hidden |> Array.collect (norm layout.OutputNormalization >> matrix layout.Output)

[<Theory>]
[<InlineData(true, false)>]
[<InlineData(false, false)>]
[<InlineData(true, true)>]
[<InlineData(false, true)>]
let ``dense forward matches independent logits with tied and separate stored outputs`` tied halfPrecision =
    use fixture = new ModelFixture(tied, halfPrecision)
    use backend = new CpuBackend() :> ITensorBackend
    let layout = fixture.Layout
    use model = TransformerModel.Load(backend, fixture, layout)
    Assert.Equal((if tied then 20 else 21), fixture.Reads)
    use state = cache backend TensorEncoding.F32
    let tokens = [| 1; 4; 2; 0 |]
    use logits = model.Forward(state, 0, ReadOnlyMemory<int>(tokens))
    Assert.Equal(TensorShape.Create [| 4; 5 |], logits.Shape)
    close (reference fixture layout tokens) (download backend logits)
    Assert.Equal((if tied then 20 else 21), fixture.Reads)
    for layer = 0 to 1 do Assert.Equal(4, state.StoredPositions layer)

[<Theory>]
[<InlineData(false)>]
[<InlineData(true)>]
let ``chunked and replaced prefixes match full forward with positional scaling`` yarn =
    use fixture = new ModelFixture(true, false)
    use backend = new CpuBackend() :> ITensorBackend
    let scaling =
        if yarn then
            ModelRopeScaling.Yarn { ScalingFactor = 4.0; OriginalContextLength = 2; BetaFast = 32.0
                                    BetaSlow = 1.0; LogMultiplier = 0.0; AttentionFactor = 1.2 }
        else ModelRopeScaling.None
    let layout = { fixture.Layout with Configuration = { fixture.Layout.Configuration with RopeScaling = scaling; AttentionTemperature = Some (AttentionTemperature.Create(0.4, 2)) } }
    use model = TransformerModel.Load(backend, fixture, layout)
    let forward state position tokens =
        use result = model.Forward(state, position, ReadOnlyMemory<int>(tokens))
        download backend result
    for encoding in [ TensorEncoding.F32; TensorEncoding.F16; TensorEncoding.BF16 ] do
        use full = cache backend encoding
        use chunks = cache backend encoding
        let expected = forward full 0 [| 1; 4; 2; 0 |]
        let first = forward chunks 0 [| 1; 4 |]
        let next = forward chunks 2 [| 2 |]
        let last = forward chunks 3 [| 0 |]
        close (Array.map float expected) (Array.concat [ first; next; last ])
        let replacement = forward chunks 1 [| 3; 1 |]
        let fresh = forward full 0 [| 1; 3; 1 |]
        close (Array.map float fresh[5..]) replacement
        for layer = 0 to 1 do Assert.Equal(3, chunks.StoredPositions layer)
    use state = cache backend TensorEncoding.F32
    close (reference fixture layout [| 1; 4; 2; 0 |]) (forward state 0 [| 1; 4; 2; 0 |])

[<Fact>]
let ``invalid forward requests leave the prefix unchanged and empty batches do not truncate`` () =
    use fixture = new ModelFixture(true, false)
    use backend = new CpuBackend() :> ITensorBackend
    use model = TransformerModel.Load(backend, fixture, fixture.Layout)
    use state = cache backend TensorEncoding.F32
    use initial = model.Forward(state, 0, ReadOnlyMemory<int>([| 1; 2 |]))
    for position, tokens in [ -1, [| 1 |]; 3, [| 1 |]; 2, Array.create 7 1; 1, [| 1; 5 |]; 1, [| -1 |] ] do
        Assert.Throws<ArgumentOutOfRangeException>(fun () -> model.Forward(state, position, ReadOnlyMemory<int>(tokens)) |> ignore) |> ignore
        for layer = 0 to 1 do Assert.Equal(2, state.StoredPositions layer)
    use empty = model.Forward(state, 0, ReadOnlyMemory<int>.Empty)
    Assert.Equal(TensorShape.Create [| 0; 5 |], empty.Shape)
    for layer = 0 to 1 do Assert.Equal(2, state.StoredPositions layer)
    use other = new CpuBackend() :> ITensorBackend
    use foreign = cache other TensorEncoding.F32
    Assert.Throws<ArgumentException>(fun () -> model.Forward(foreign, 0, ReadOnlyMemory<int>([| 1 |])) |> ignore) |> ignore
    use wrong = backend.AllocateKeyValueCache(KeyValueCacheLayout.Create(1, 8, 2, 2, TensorEncoding.F32, TensorEncoding.F32))
    Assert.Throws<ArgumentException>(fun () -> model.Forward(wrong, 0, ReadOnlyMemory<int>([| 1 |])) |> ignore) |> ignore
    use rows = backend.Allocate(TensorShape.Create [| 0; 4 |])
    backend.StoreKeyValues(state, 1, 0, rows, rows)
    Assert.Throws<ArgumentException>(fun () -> model.Forward(state, 0, ReadOnlyMemory<int>([| 1 |])) |> ignore) |> ignore

[<Fact>]
let ``loading rejects malformed public layouts before reading payloads`` () =
    use fixture = new ModelFixture(true, false)
    use backend = new CpuBackend() :> ITensorBackend
    let original = fixture.Layout
    let configurations =
        [ { original.Configuration with BlockCount = 0 }
          { original.Configuration with KeyLength = 3 }
          { original.Configuration with ValueLength = 4 }
          { original.Configuration with AttentionHeadCount = 3 }
          { original.Configuration with AttentionHeadCount = Int32.MaxValue - 1 }
          { original.Configuration with RmsNormalizationEpsilon = Single.NaN } ]
    for configuration in configurations do
        Assert.Throws<InvalidDataException>(fun () -> TransformerModel.Load(backend, fixture, { original with Configuration = configuration }) |> ignore) |> ignore
    for embedding in
        [ { original.TokenEmbedding with Shape = [| 5UL; 8UL |] }
          { original.TokenEmbedding with Encoding = None }
          { original.TokenEmbedding with ByteSize = Some 1UL }
          { original.TokenEmbedding with Name = "missing" } ] do
        Assert.Throws<InvalidDataException>(fun () -> TransformerModel.Load(backend, fixture, { original with TokenEmbedding = embedding }) |> ignore) |> ignore
    Assert.Equal(0, fixture.Reads)

[<Fact>]
let ``model disposal preserves borrowed backend source and returned logits`` () =
    use fixture = new ModelFixture(true, false)
    use backend = new CpuBackend() :> ITensorBackend
    use model = TransformerModel.Load(backend, fixture, fixture.Layout)
    use state = cache backend TensorEncoding.F32
    use logits = model.Forward(state, 0, ReadOnlyMemory<int>([| 2 |]))
    (model :> IDisposable).Dispose()
    (model :> IDisposable).Dispose()
    close (reference fixture fixture.Layout [| 2 |]) (download backend logits)
    Assert.Throws<ObjectDisposedException>(fun () -> model.Forward(state, 1, ReadOnlyMemory<int>([| 2 |])) |> ignore) |> ignore
    use second = TransformerModel.Load(backend, fixture, fixture.Layout)
    backend.Synchronize()

[<Fact>]
let ``load failures propagate and leave dependencies usable`` () =
    use fixture = new ModelFixture(true, false)
    use backend = new CpuBackend() :> ITensorBackend
    fixture.FailName <- "layer1.down"
    Assert.Throws<IOException>(fun () -> TransformerModel.Load(backend, fixture, fixture.Layout) |> ignore) |> ignore
    fixture.FailName <- ""
    use model = TransformerModel.Load(backend, fixture, fixture.Layout)
    use state = cache backend TensorEncoding.F32
    use result = model.Forward(state, 0, ReadOnlyMemory<int>([| 1 |]))
    close (reference fixture fixture.Layout [| 1 |]) (download backend result)


[<Fact>]
let ``forward matches independent logits for Q8_0 matrices`` () =
    use fixture = new ModelFixture(true, false, quantized = true)
    use backend = new CpuBackend() :> ITensorBackend
    use model = TransformerModel.Load(backend, fixture, fixture.Layout)
    use state = cache backend TensorEncoding.F32
    let tokens = [| 1; 3; 2 |]
    use result = model.Forward(state, 0, ReadOnlyMemory<int>(tokens))
    close (reference fixture fixture.Layout tokens) (download backend result)

// Delegate numerical work to CPU while recording ownership and injecting an execution failure.
type private TrackingBackend() =
    let inner = new CpuBackend() :> ITensorBackend
    let tensors = ResizeArray<ITensor>()
    let weights = ResizeArray<IWeightTensor>()
    let caches = ResizeArray<IKeyValueCache>()
    let batches = ResizeArray<int>()
    let mutable onSynchronize = ignore
    let mutable products = 0
    let mutable failProduct = -1
    let track tensor =
        tensors.Add tensor
        tensor
    member _.Tensors = tensors.ToArray()
    member _.Weights = weights.ToArray()
    member _.FailProduct with set value = failProduct <- value
    member _.Caches = caches.ToArray()
    member _.Batches = batches.ToArray()
    member _.OnSynchronize with set action = onSynchronize <- action
    interface ITensorBackend with
        member _.Backend = inner.Backend
        member _.Allocate shape = inner.Allocate shape |> track
        member _.Upload(shape, data) = inner.Upload(shape, data) |> track
        member _.UploadWeights matrix =
            let weight = inner.UploadWeights matrix
            weights.Add weight
            weight
        member _.AllocateKeyValueCache layout =
            let cache = inner.AllocateKeyValueCache layout
            caches.Add cache
            cache
        member _.Download(tensor, data) = inner.Download(tensor, data)
        member _.Synchronize() =
            inner.Synchronize()
            onSynchronize ()
    interface IAddKernel with
        member _.Add(a, b, output) = inner.Add(a, b, output)
    interface ISwiGluKernel with
        member _.SwiGlu(a, b, output) = inner.SwiGlu(a, b, output)
    interface IRmsNormKernel with
        member _.RmsNorm(input, weight, epsilon, output) = inner.RmsNorm(input, weight, epsilon, output)
    interface IEmbeddingKernel with
        member _.EmbeddingLookup(weight, ids, output) =
            batches.Add ids.Length
            inner.EmbeddingLookup(weight, ids, output)
    interface IMatVecKernel with
        member _.MatVec(weight, input, output) = inner.MatVec(weight, input, output)
    interface IMatMulKernel with
        member _.MatMul(weight, input, output) =
            products <- products + 1
            if products = failProduct then raise (InvalidOperationException("Injected matrix product failure."))
            inner.MatMul(weight, input, output)
    interface IRopeKernel with
        member _.Rope(input, positions, rope, output) = inner.Rope(input, positions, rope, output)
    interface IAttentionTemperatureKernel with
        member _.ApplyAttentionTemperature(input, positions, temperature, output) = inner.ApplyAttentionTemperature(input, positions, temperature, output)
    interface IAttentionKernel with
        member _.Attention(cache, layer, position, queries, output) = inner.Attention(cache, layer, position, queries, output)
    interface IKeyValueCacheKernel with
        member _.StoreKeyValues(cache, layer, position, keys, values) = inner.StoreKeyValues(cache, layer, position, keys, values)
        member _.ReadKeyValues(cache, layer, position, keys, values) = inner.ReadKeyValues(cache, layer, position, keys, values)
    interface IDisposable with
        member _.Dispose() = inner.Dispose()

[<Fact>]
let ``failed loading disposes every uploaded weight and normalization tensor`` () =
    use fixture = new ModelFixture(true, false)
    use tracking = new TrackingBackend()
    let backend = tracking :> ITensorBackend
    fixture.FailName <- "layer1.down"
    Assert.Throws<IOException>(fun () -> TransformerModel.Load(backend, fixture, fixture.Layout) |> ignore) |> ignore
    Assert.NotEmpty(tracking.Weights)
    Assert.NotEmpty(tracking.Tensors)
    for tensor in tracking.Tensors do
        Assert.Throws<ObjectDisposedException>(fun () -> download backend tensor |> ignore) |> ignore
    for weight in tracking.Weights do
        use destination = backend.Allocate(TensorShape.Create [| 0; weight.Columns |])
        Assert.Throws<ObjectDisposedException>(fun () -> backend.EmbeddingLookup(weight, ReadOnlyMemory<int>.Empty, destination)) |> ignore

[<Fact>]
let ``execution failure releases scratch and logits while leaving the partial cache explicit`` () =
    use fixture = new ModelFixture(true, false)
    use tracking = new TrackingBackend()
    let backend = tracking :> ITensorBackend
    use model = TransformerModel.Load(backend, fixture, fixture.Layout)
    let persistentCount = tracking.Tensors.Length
    use state = cache backend TensorEncoding.F32
    // Seven products in the first block; fail while preparing the second block's keys.
    tracking.FailProduct <- 9
    Assert.Throws<InvalidOperationException>(fun () -> model.Forward(state, 0, ReadOnlyMemory<int>([| 1 |])) |> ignore) |> ignore
    Assert.Equal(1, state.StoredPositions 0)
    Assert.Equal(0, state.StoredPositions 1)
    for tensor in tracking.Tensors |> Array.skip persistentCount do
        Assert.Throws<ObjectDisposedException>(fun () -> download backend tensor |> ignore) |> ignore
    Assert.Throws<ArgumentException>(fun () -> model.Forward(state, 0, ReadOnlyMemory<int>([| 1 |])) |> ignore) |> ignore
    use replacement = cache backend TensorEncoding.F32
    use result = model.Forward(replacement, 0, ReadOnlyMemory<int>([| 1 |]))
    close (reference fixture fixture.Layout [| 1 |]) (download backend result)

let private session model capacity batchSize =
    GenerationSession.Create(model, capacity, batchSize, TensorEncoding.F32, TensorEncoding.F32)

[<Theory>]
[<InlineData(1)>]
[<InlineData(2)>]
[<InlineData(3)>]
[<InlineData(8)>]
let ``session prefill and decoding match the full scalar reference for every batch size`` batchSize =
    use fixture = new ModelFixture(false, false)
    use tracking = new TrackingBackend()
    let layout =
        { fixture.Layout with
            Configuration = { fixture.Layout.Configuration with AttentionTemperature = Some (AttentionTemperature.Create(0.3, 2)) } }
    use model = TransformerModel.Load(tracking, fixture, layout)
    use sequence = session model 8 batchSize
    let prompt = [| 1; 4; 2; 0; 3 |]
    let logits = sequence.Prefill(ReadOnlyMemory<int>(prompt))
    close (reference fixture layout prompt |> Array.skip 20) logits
    Assert.Equal(5, sequence.ProcessedTokenCount)
    Assert.All(tracking.Batches, fun count -> Assert.InRange(count, 1, batchSize))
    Assert.Equal(5, Array.sum tracking.Batches)
    // Returned logits are caller-owned; modifying them cannot affect later execution.
    Array.fill logits 0 logits.Length Single.NaN
    let next = sequence.Decode(2)
    close (reference fixture layout (Array.append prompt [| 2 |]) |> Array.skip 25) next
    let final = sequence.Decode(1)
    close (reference fixture layout (Array.append prompt [| 2; 1 |]) |> Array.skip 30) final
    Assert.Equal(7, sequence.ProcessedTokenCount)
    Assert.Equal<int>([| 1; 1 |], tracking.Batches |> Array.rev |> Array.take 2)

[<Fact>]
let ``sessions share model weights while keeping independent sequences and resetting their caches`` () =
    use fixture = new ModelFixture(true, false)
    use backend = new CpuBackend() :> ITensorBackend
    use model = TransformerModel.Load(backend, fixture, fixture.Layout)
    use first = session model 8 2
    use second = session model 8 3
    let firstLogits = first.Prefill(ReadOnlyMemory<int>([| 1; 2; 3 |]))
    let secondLogits = second.Prefill(ReadOnlyMemory<int>([| 4 |]))
    close (reference fixture fixture.Layout [| 1; 2; 3 |] |> Array.skip 10) firstLogits
    close (reference fixture fixture.Layout [| 4 |]) secondLogits
    close (reference fixture fixture.Layout [| 1; 2; 3; 0 |] |> Array.skip 15) (first.Decode 0)
    close (reference fixture fixture.Layout [| 4; 2 |] |> Array.skip 5) (second.Decode 2)
    first.Reset()
    Assert.Equal(0, first.ProcessedTokenCount)
    close (reference fixture fixture.Layout [| 0; 1 |] |> Array.skip 5) (first.Prefill(ReadOnlyMemory<int>([| 0; 1 |])))
    Assert.Equal(20, fixture.Reads)

[<Fact>]
let ``session validates the entire prompt and available capacity before changing the sequence`` () =
    use fixture = new ModelFixture(true, false)
    use tracking = new TrackingBackend()
    use model = TransformerModel.Load(tracking, fixture, fixture.Layout)
    use sequence = session model 3 1
    Assert.Throws<InvalidOperationException>(fun () -> sequence.Decode(0) |> ignore) |> ignore
    Assert.Throws<ArgumentException>(fun () -> sequence.Prefill(ReadOnlyMemory<int>.Empty) |> ignore) |> ignore
    for prompt in [ [| 0; 1; 5 |]; [| 0; -1 |]; [| 0; 1; 2; 3 |] ] do
        Assert.Throws<ArgumentOutOfRangeException>(fun () -> sequence.Prefill(ReadOnlyMemory<int>(prompt)) |> ignore) |> ignore
    Assert.Empty(tracking.Batches)
    Assert.Equal(0, sequence.ProcessedTokenCount)
    Assert.False(sequence.IsFaulted)
    sequence.Prefill(ReadOnlyMemory<int>([| 1; 2 |])) |> ignore
    Assert.Throws<InvalidOperationException>(fun () -> sequence.Prefill(ReadOnlyMemory<int>([| 1 |])) |> ignore) |> ignore
    Assert.Throws<ArgumentOutOfRangeException>(fun () -> sequence.Decode(5) |> ignore) |> ignore
    Assert.Equal(2, sequence.ProcessedTokenCount)
    sequence.Decode(0) |> ignore
    Assert.Throws<ArgumentOutOfRangeException>(fun () -> sequence.Decode(1) |> ignore) |> ignore
    Assert.Equal(3, sequence.ProcessedTokenCount)
    Assert.False(sequence.IsFaulted)
    sequence.Reset()
    sequence.Prefill(ReadOnlyMemory<int>([| 0; 1; 2 |])) |> ignore
    Assert.Equal(3, sequence.ProcessedTokenCount)

[<Fact>]
let ``session cancellation before execution is harmless and cancellation between batches faults the prefix`` () =
    use fixture = new ModelFixture(true, false)
    use tracking = new TrackingBackend()
    use model = TransformerModel.Load(tracking, fixture, fixture.Layout)
    use sequence = session model 8 2
    use cancellation = new System.Threading.CancellationTokenSource()
    cancellation.Cancel()
    Assert.Throws<OperationCanceledException>(fun () -> sequence.Prefill(ReadOnlyMemory<int>([| 1 |]), cancellation.Token) |> ignore) |> ignore
    Assert.False(sequence.IsFaulted)
    Assert.Empty(tracking.Batches)
    use between = new System.Threading.CancellationTokenSource()
    tracking.OnSynchronize <- fun () -> between.Cancel()
    Assert.Throws<OperationCanceledException>(fun () -> sequence.Prefill(ReadOnlyMemory<int>([| 1; 2; 3 |]), between.Token) |> ignore) |> ignore
    Assert.True(sequence.IsFaulted)
    Assert.Equal(2, sequence.ProcessedTokenCount)
    Assert.Throws<InvalidOperationException>(fun () -> sequence.Decode(0) |> ignore) |> ignore
    tracking.OnSynchronize <- ignore
    sequence.Reset()
    Assert.False(sequence.IsFaulted)
    sequence.Prefill(ReadOnlyMemory<int>([| 1 |])) |> ignore
    Assert.Throws<OperationCanceledException>(fun () -> sequence.Decode(0, cancellation.Token) |> ignore) |> ignore
    Assert.Equal(1, sequence.ProcessedTokenCount)
    Assert.False(sequence.IsFaulted)
    close (reference fixture fixture.Layout [| 1; 0 |] |> Array.skip 5) (sequence.Decode 0)

[<Theory>]
[<InlineData(false)>]
[<InlineData(true)>]
let ``session execution errors require reset even when they happen during single token decoding`` decoding =
    use fixture = new ModelFixture(true, false)
    use tracking = new TrackingBackend()
    use model = TransformerModel.Load(tracking, fixture, fixture.Layout)
    use sequence = session model 8 2
    if decoding then sequence.Prefill(ReadOnlyMemory<int>([| 1 |])) |> ignore
    tracking.FailProduct <- if decoding then 24 else 9
    let operation () =
        if decoding then sequence.Decode(2) |> ignore
        else sequence.Prefill(ReadOnlyMemory<int>([| 1; 2; 3 |])) |> ignore
    Assert.Throws<InvalidOperationException>(Action operation) |> ignore
    Assert.True(sequence.IsFaulted)
    Assert.Equal((if decoding then 1 else 0), sequence.ProcessedTokenCount)
    Assert.Throws<InvalidOperationException>(fun () -> sequence.Prefill(ReadOnlyMemory<int>([| 0 |])) |> ignore) |> ignore
    sequence.Reset()
    close (reference fixture fixture.Layout [| 2 |]) (sequence.Prefill(ReadOnlyMemory<int>([| 2 |])))

[<Fact>]
let ``session owns and releases its caches but borrows the model`` () =
    use fixture = new ModelFixture(true, false)
    use tracking = new TrackingBackend()
    use model = TransformerModel.Load(tracking, fixture, fixture.Layout)
    use sequence = session model 8 2
    sequence.Prefill(ReadOnlyMemory<int>([| 1; 2 |])) |> ignore
    sequence.Reset()
    Assert.Throws<ObjectDisposedException>(fun () -> tracking.Caches[0].StoredPositions(0) |> ignore) |> ignore
    (sequence :> IDisposable).Dispose()
    (sequence :> IDisposable).Dispose()
    for cache in tracking.Caches do
        Assert.Throws<ObjectDisposedException>(fun () -> cache.StoredPositions(0) |> ignore) |> ignore
    Assert.Throws<ObjectDisposedException>(fun () -> sequence.Reset()) |> ignore
    Assert.Throws<ObjectDisposedException>(fun () -> sequence.Prefill(ReadOnlyMemory<int>([| 0 |])) |> ignore) |> ignore
    Assert.Throws<ObjectDisposedException>(fun () -> sequence.Decode(0) |> ignore) |> ignore
    use next = session model 8 2
    next.Prefill(ReadOnlyMemory<int>([| 0 |])) |> ignore
    (model :> IDisposable).Dispose()
    Assert.Throws<ObjectDisposedException>(fun () -> next.Decode(1) |> ignore) |> ignore
    Assert.Throws<ObjectDisposedException>(fun () -> session model 8 2 |> ignore) |> ignore

[<Fact>]
let ``session creation rejects invalid limits and incompatible cache encodings`` () =
    use fixture = new ModelFixture(true, false)
    use backend = new CpuBackend() :> ITensorBackend
    use model = TransformerModel.Load(backend, fixture, fixture.Layout)
    for capacity, batchSize in [ 0, 2; -1, 2; 9, 2; 8, 0; 8, -1 ] do
        Assert.Throws<ArgumentOutOfRangeException>(fun () -> session model capacity batchSize |> ignore) |> ignore
    Assert.Throws<ArgumentException>(fun () -> GenerationSession.Create(model, 8, 2, TensorEncoding.Q8_0, TensorEncoding.F32) |> ignore) |> ignore
    Assert.Throws<ArgumentNullException>(fun () -> session Unchecked.defaultof<_> 8 2 |> ignore) |> ignore
