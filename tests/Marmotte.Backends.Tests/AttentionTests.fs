module Marmotte.Backends.AttentionTests

open System
open Marmotte.Core
open Marmotte.Cpu
open Marmotte.Cuda
open Marmotte.Metal
open Xunit

let private upload (backend: ITensorBackend) shape (values: single array) =
    backend.Upload(TensorShape.Create shape, ReadOnlyMemory<single>(values))

let private download (backend: ITensorBackend) (tensor: ITensor) =
    let values = Array.zeroCreate<single> tensor.Shape.ElementCount
    backend.Download(tensor, values.AsMemory())
    values

let private close (expected: single array) (actual: single array) =
    Assert.Equal(expected.Length, actual.Length)
    Array.iter2 (fun expected actual ->
        Assert.True(abs (expected - actual) < 0.00002f, $"Expected {expected}, got {actual}.")) expected actual

// Independent dense reference: expand the KV head mapping and reduce in double precision.
let private reference dimension kvHeads queryHeads position tokens (keys: single array) (values: single array) (queries: single array) =
    Array.init (tokens * queryHeads * dimension) (fun index ->
        let channel = index % dimension
        let head = (index / dimension) % queryHeads
        let token = index / (dimension * queryHeads)
        let kvHead = head / (queryHeads / kvHeads)
        let scores =
            [| for p in 0 .. position + token ->
                [| for c in 0 .. dimension - 1 ->
                    float queries[(token * queryHeads + head) * dimension + c]
                    * float keys[(p * kvHeads + kvHead) * dimension + c] |]
                |> Array.sum
                |> fun score -> score / sqrt (float dimension) |]
        let maximum = Array.max scores
        let weights = scores |> Array.map (fun score -> exp (score - maximum))
        let numerator = weights |> Array.mapi (fun p weight -> weight * float values[(p * kvHeads + kvHead) * dimension + channel]) |> Array.sum
        single (numerator / Array.sum weights))

[<Theory>]
[<InlineData(1)>]
[<InlineData(2)>]
[<InlineData(8)>]
let ``attention matches a dense reference for multi-query grouped and multi-head layouts`` kvHeads =
    use backend = new CpuBackend() :> ITensorBackend
    let dimension, queryHeads, tokens = 4, 8, 4
    use cache = backend.AllocateKeyValueCache(KeyValueCacheLayout.Create(2, 6, kvHeads, dimension, TensorEncoding.F32, TensorEncoding.F32))
    let random = Random(609)
    let randomValues count = Array.init count (fun _ -> single (random.NextDouble() * 4.0 - 2.0))
    let keys, values = randomValues (tokens * kvHeads * dimension), randomValues (tokens * kvHeads * dimension)
    let queries = randomValues (tokens * queryHeads * dimension)
    use k = upload backend [| tokens; kvHeads; dimension |] keys
    use v = upload backend [| tokens; kvHeads; dimension |] values
    backend.StoreKeyValues(cache, 1, 0, k, v)
    use q = upload backend [| tokens; queryHeads; dimension |] queries
    use output = backend.Allocate(q.Shape)
    backend.Attention(cache, 1, 0, q, output)
    let batch = download backend output
    close (reference dimension kvHeads queryHeads 0 tokens keys values queries) batch
    Assert.Equal<single>(queries, download backend q)
    for p = 0 to tokens - 1 do
        use query = upload backend [| 1; queryHeads * dimension |] queries[p * 32 .. p * 32 + 31]
        use result = backend.Allocate(query.Shape)
        backend.Attention(cache, 1, p, query, result)
        close batch[p * 32 .. p * 32 + 31] (download backend result)
    use suffix = upload backend [| 2; 32 |] queries[64..]
    use suffixResult = backend.Allocate(suffix.Shape)
    backend.Attention(cache, 1, 2, suffix, suffixResult)
    close batch[64..] (download backend suffixResult)

[<Fact>]
let ``every cache encoding and mixed keys and values match attention on decoded rows`` () =
    let encodings = [ TensorEncoding.F32; TensorEncoding.F16; TensorEncoding.BF16; TensorEncoding.Q4_0; TensorEncoding.Q8_0; TensorEncoding.Q4_1; TensorEncoding.Q5_0; TensorEncoding.Q5_1; TensorEncoding.IQ4_NL ]
    use backend = new CpuBackend() :> ITensorBackend
    for keyEncoding in encodings do
        for valueEncoding in encodings do
            use cache = backend.AllocateKeyValueCache(KeyValueCacheLayout.Create(1, 3, 2, 32, keyEncoding, valueEncoding))
            let data phase count = Array.init count (fun i -> single (sin (float i + phase)))
            use keys = upload backend [| 3; 64 |] (data 0.0 192)
            use values = upload backend [| 3; 64 |] (data 1.0 192)
            backend.StoreKeyValues(cache, 0, 0, keys, values)
            use decodedKeys = backend.Allocate(keys.Shape)
            use decodedValues = backend.Allocate(values.Shape)
            backend.ReadKeyValues(cache, 0, 0, decodedKeys, decodedValues)
            let queries = data 2.0 384
            use q = upload backend [| 3; 4; 32 |] queries
            use result = backend.Allocate(q.Shape)
            backend.Attention(cache, 0, 0, q, result)
            close (reference 32 2 4 0 3 (download backend decodedKeys) (download backend decodedValues) queries) (download backend result)

[<Fact>]
let ``softmax stays finite for large scores and never reads masked nonfinite rows`` () =
    use backend = new CpuBackend() :> ITensorBackend
    use cache = backend.AllocateKeyValueCache(KeyValueCacheLayout.Create(1, 3, 1, 1, TensorEncoding.F32, TensorEncoding.F32))
    use keys = upload backend [| 3; 1 |] [| 1000.0f; 1001.0f; Single.NaN |]
    use values = upload backend [| 3; 1 |] [| 2.0f; 6.0f; Single.NaN |]
    backend.StoreKeyValues(cache, 0, 0, keys, values)
    use q = upload backend [| 2; 1 |] [| 1.0f; 1.0f |]
    use result = backend.Allocate(q.Shape)
    backend.Attention(cache, 0, 0, q, result)
    close [| 2.0f; single ((2.0 + 6.0 * exp 1.0) / (1.0 + exp 1.0)) |] (download backend result)

[<Fact>]
let ``validation rejects incompatible arguments before writing and accepts empty batches`` () =
    use backend = new CpuBackend() :> ITensorBackend
    use other = new CpuBackend() :> ITensorBackend
    let layout = KeyValueCacheLayout.Create(2, 3, 2, 2, TensorEncoding.F32, TensorEncoding.F32)
    use cache = backend.AllocateKeyValueCache(layout)
    use rows = upload backend [| 2; 4 |] (Array.zeroCreate 8)
    backend.StoreKeyValues(cache, 0, 0, rows, rows)
    use q = upload backend [| 2; 8 |] (Array.zeroCreate 16)
    use result = upload backend [| 2; 8 |] (Array.create 16 42.0f)
    for layer, position in [ -1, 0; 2, 0; 1, 0; 0, -1; 0, 1; 0, Int32.MaxValue ] do
        Assert.Throws<ArgumentOutOfRangeException>(fun () -> backend.Attention(cache, layer, position, q, result)) |> ignore
    for shape in [ [| 16 |]; [| 2; 3; 2 |]; [| 2; 2; 4 |]; [| 2; 0; 2 |]; [| 2; 7 |] ] do
        use bad = backend.Allocate(TensorShape.Create shape)
        Assert.Throws<ArgumentException>(fun () -> backend.Attention(cache, 0, 0, bad, result)) |> ignore
    use mismatch = backend.Allocate(TensorShape.Create [| 2; 4; 2 |])
    Assert.Throws<ArgumentException>(fun () -> backend.Attention(cache, 0, 0, q, mismatch)) |> ignore
    Assert.Throws<ArgumentException>(fun () -> backend.Attention(cache, 0, 0, q, q)) |> ignore
    use foreign = other.Allocate(q.Shape)
    use foreignCache = other.AllocateKeyValueCache(layout)
    Assert.Throws<ArgumentException>(fun () -> backend.Attention(foreignCache, 0, 0, q, result)) |> ignore
    Assert.Throws<ArgumentException>(fun () -> backend.Attention(cache, 0, 0, foreign, result)) |> ignore
    Assert.Throws<ArgumentException>(fun () -> backend.Attention(cache, 0, 0, q, foreign)) |> ignore
    Assert.Throws<ArgumentNullException>(fun () -> backend.Attention(Unchecked.defaultof<_>, 0, 0, q, result)) |> ignore
    Assert.Throws<ArgumentNullException>(fun () -> backend.Attention(cache, 0, 0, Unchecked.defaultof<_>, result)) |> ignore
    Assert.Throws<ArgumentNullException>(fun () -> backend.Attention(cache, 0, 0, q, Unchecked.defaultof<_>)) |> ignore
    Assert.Equal<single>(Array.create 16 42.0f, download backend result)
    use empty = backend.Allocate(TensorShape.Create [| 0; 4; 2 |])
    use emptyResult = backend.Allocate(empty.Shape)
    backend.Attention(cache, 0, 2, empty, emptyResult)
    backend.Attention(cache, 1, 0, empty, emptyResult)
    use emptyCacheRows = backend.Allocate(TensorShape.Create [| 0; 4 |])
    backend.StoreKeyValues(cache, 0, 0, emptyCacheRows, emptyCacheRows)
    Assert.Throws<ArgumentOutOfRangeException>(fun () -> backend.Attention(cache, 0, 0, q, result)) |> ignore
    cache.Dispose()
    Assert.Throws<ObjectDisposedException>(fun () -> backend.Attention(cache, 0, 0, q, result)) |> ignore

[<Fact>]
let ``attention respects tensor and backend disposal`` () =
    use backend = new CpuBackend() :> ITensorBackend
    use cache = backend.AllocateKeyValueCache(KeyValueCacheLayout.Create(1, 1, 1, 1, TensorEncoding.F32, TensorEncoding.F32))
    use q = upload backend [| 1; 1 |] [| 0.0f |]
    use result = backend.Allocate(q.Shape)
    backend.StoreKeyValues(cache, 0, 0, q, q)
    result.Dispose()
    Assert.Throws<ObjectDisposedException>(fun () -> backend.Attention(cache, 0, 0, q, result)) |> ignore
    use liveResult = backend.Allocate(q.Shape)
    q.Dispose()
    Assert.Throws<ObjectDisposedException>(fun () -> backend.Attention(cache, 0, 0, q, liveResult)) |> ignore
    backend.Dispose()
    Assert.Throws<ObjectDisposedException>(fun () -> backend.Attention(cache, 0, 0, q, liveResult)) |> ignore

[<Fact>]
let ``GPU attention placeholders fail explicitly without drivers`` () =
    for backend in [ new CudaBackend() :> ITensorBackend; new MetalBackend() :> ITensorBackend ] do
        use backend = backend
        let attention () = backend.Attention(Unchecked.defaultof<_>, 0, 0, Unchecked.defaultof<_>, Unchecked.defaultof<_>)
        Assert.Throws<NotImplementedException>(Action attention) |> ignore
        backend.Dispose()
        Assert.Throws<ObjectDisposedException>(Action attention) |> ignore

[<Theory>]
[<InlineData(1)>]
[<InlineData(7)>]
[<InlineData(32)>]
[<InlineData(128)>]
let ``online attention matches score buffer reference for long changing maxima`` dimension =
    let tokens = 257
    let keys = Array.init (tokens * dimension) (fun i -> single (i / dimension % 23 - 11) / single dimension)
    let values = Array.init (tokens * dimension) (fun i -> single (sin (float i * 0.13)))
    let queries = Array.create (3 * 4 * dimension) 7.0f
    let execute mode =
        use backend = new CpuBackend(CpuMatrixProductMode.DecodedFloat, mode) :> ITensorBackend
        use state = backend.AllocateKeyValueCache(KeyValueCacheLayout.Create(1, tokens, 1, dimension, TensorEncoding.F32, TensorEncoding.F32))
        use k = upload backend [| tokens; dimension |] keys
        use v = upload backend [| tokens; dimension |] values
        backend.StoreKeyValues(state, 0, 0, k, v)
        use q = upload backend [| 3; 4 * dimension |] queries
        use output = backend.Allocate(q.Shape)
        backend.Attention(state, 0, tokens - 3, q, output)
        download backend output
    close (execute CpuAttentionMode.Reference) (execute CpuAttentionMode.Online)

[<Fact>]
let ``online attention preserves nonfinite score behavior including initial negative infinity`` () =
    for scores in [ [| Single.NegativeInfinity; 1.0f |]
                    [| Single.NegativeInfinity; Single.NegativeInfinity |]
                    [| Single.PositiveInfinity; 1.0f |]
                    [| 1.0f; Single.NaN |] ] do
        let execute mode =
            use backend = new CpuBackend(CpuMatrixProductMode.DecodedFloat, mode) :> ITensorBackend
            use state = backend.AllocateKeyValueCache(KeyValueCacheLayout.Create(1, 2, 1, 1, TensorEncoding.F32, TensorEncoding.F32))
            use k = upload backend [| 2; 1 |] scores
            use v = upload backend [| 2; 1 |] [| 2.0f; 3.0f |]
            backend.StoreKeyValues(state, 0, 0, k, v)
            use q = upload backend [| 1; 1 |] [| 1.0f |]
            use output = backend.Allocate(q.Shape)
            backend.Attention(state, 0, 1, q, output)
            (download backend output)[0]
        let expected = execute CpuAttentionMode.Reference
        let actual = execute CpuAttentionMode.Online
        if Single.IsNaN expected then Assert.True(Single.IsNaN actual)
        else Assert.Equal(expected, actual)

[<Fact>]
let ``online numerator does not overflow for a finite large weighted average`` () =
    use backend = new CpuBackend() :> ITensorBackend
    use state = backend.AllocateKeyValueCache(KeyValueCacheLayout.Create(1, 64, 1, 4, TensorEncoding.F32, TensorEncoding.F32))
    use keys = upload backend [| 64; 4 |] (Array.zeroCreate 256)
    use values = upload backend [| 64; 4 |] (Array.create 256 1e38f)
    backend.StoreKeyValues(state, 0, 0, keys, values)
    use queries = upload backend [| 1; 4 |] (Array.create 4 1.0f)
    use result = backend.Allocate(queries.Shape)
    backend.Attention(state, 0, 63, queries, result)
    Assert.Equal<single>(Array.create 4 1e38f, download backend result)
