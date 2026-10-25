module Marmotte.Backends.KeyValueCacheTests

open System
open Marmotte.Core
open Marmotte.Cpu
open Xunit

let private cpu () = new CpuBackend() :> ITensorBackend

let private download (backend: ITensorBackend) (tensor: ITensor) =
    let values = Array.zeroCreate<single> tensor.Shape.ElementCount
    backend.Download(tensor, values.AsMemory())
    values

let private upload (backend: ITensorBackend) (dimensions: int array) (values: single array) =
    backend.Upload(TensorShape.Create dimensions, ReadOnlyMemory<single>(values))

/// Two layers of 4 positions, each with 2 heads of 32 values: one Q8_0 or Q4_0 block per head.
let private layout keyEncoding valueEncoding =
    KeyValueCacheLayout.Create(
        layers = 2,
        capacity = 4,
        keyValueHeads = 2,
        headDimension = 32,
        keyEncoding = keyEncoding,
        valueEncoding = valueEncoding)

/// Reads positions [position, position + tokens) of a layer back as host keys and values.
let private read (backend: ITensorBackend) (cache: IKeyValueCache) layer position tokens =
    use keys = backend.Allocate(TensorShape.Create [| tokens; 64 |])
    use values = backend.Allocate(TensorShape.Create [| tokens; 64 |])
    backend.ReadKeyValues(cache, layer, position, keys, values)
    download backend keys, download backend values

let private randomValues (random: Random) count =
    Array.init count (fun _ -> float32 (random.NextDouble() * 4.0 - 2.0))

[<Fact>]
let ``an F32 cache appends and reads back positions of each layer bit for bit`` () =
    use backend = cpu ()
    use cache = backend.AllocateKeyValueCache(layout TensorEncoding.F32 TensorEncoding.F32)
    let random = Random(5051)
    let promptKeys = randomValues random (3 * 64)
    let promptValues = randomValues random (3 * 64)
    let nextKeys = randomValues random 64
    let nextValues = Array.append [| -0.0f; Single.PositiveInfinity |] (randomValues random 62)

    // Prompt processing stores three positions; generation then appends one, here as [1; 2; 32].
    use keys = upload backend [| 3; 64 |] promptKeys
    use values = upload backend [| 3; 64 |] promptValues
    backend.StoreKeyValues(cache, 1, 0, keys, values)
    use nextKeyTensor = upload backend [| 1; 2; 32 |] nextKeys
    use nextValueTensor = upload backend [| 1; 2; 32 |] nextValues
    backend.StoreKeyValues(cache, 1, 3, nextKeyTensor, nextValueTensor)

    Assert.Equal(0, cache.StoredPositions 0)
    Assert.Equal(4, cache.StoredPositions 1)
    Assert.Equal<single>(promptKeys, download backend keys)

    let storedKeys, storedValues = read backend cache 1 0 4
    let bits (values: single array) = values |> Array.map BitConverter.SingleToUInt32Bits
    Assert.Equal<uint32>(bits (Array.append promptKeys nextKeys), bits storedKeys)
    Assert.Equal<uint32>(bits (Array.append promptValues nextValues), bits storedValues)

    let lastKeys, _ = read backend cache 1 3 1
    Assert.Equal<single>(nextKeys, lastKeys)
    let emptyKeys, emptyValues = read backend cache 0 0 0
    Assert.Empty(emptyKeys)
    Assert.Empty(emptyValues)

[<Fact>]
let ``keys and values are stored in their own quantized encodings`` () =
    use backend = cpu ()
    use cache = backend.AllocateKeyValueCache(layout TensorEncoding.Q8_0 TensorEncoding.Q4_0)

    // Each head is one block. Integers from -127 to 127 starting with 127 give Q8_0 scale 1, and
    // the values j - 8 then 7 - j give Q4_0 scale 1 (see RowEncoderTests): both decode exactly.
    let exactKeys = Array.init 64 (fun index -> float32 (if index % 32 = 0 then 127 else (index * 37) % 255 - 127))
    let exactValues = Array.init 64 (fun index -> let lane = index % 32 in float32 (if lane < 16 then lane - 8 else 23 - lane))
    use keys = upload backend [| 1; 64 |] exactKeys
    use values = upload backend [| 1; 64 |] exactValues
    backend.StoreKeyValues(cache, 0, 0, keys, values)

    let storedKeys, storedValues = read backend cache 0 0 1
    Assert.Equal<single>(exactKeys, storedKeys)
    Assert.Equal<single>(exactValues, storedValues)

    // The same tensor stored as keys and values keeps about 8 bits per value in Q8_0, 4 in Q4_0.
    let shared = randomValues (Random(5253)) 64
    use sharedTensor = upload backend [| 1; 64 |] shared
    backend.StoreKeyValues(cache, 0, 1, sharedTensor, sharedTensor)
    let sharedKeys, sharedValues = read backend cache 0 1 1
    let largestError (decoded: single array) = Array.map2 (fun a b -> abs (a - b)) decoded shared |> Array.max
    let maximum = shared |> Array.map abs |> Array.max
    Assert.InRange(largestError sharedKeys, 0.0f, maximum * (1.0f / 254.0f + 0.00048828125f))
    Assert.InRange(largestError sharedValues, 0.0f, maximum * (1.0f / 8.0f + 0.00048828125f))
    Assert.True(largestError sharedKeys < largestError sharedValues)

[<Fact>]
let ``storing before the end replaces positions and discards the later ones`` () =
    use backend = cpu ()
    use cache = backend.AllocateKeyValueCache(layout TensorEncoding.F16 TensorEncoding.F16)
    let first = Array.init (4 * 64) (fun index -> float32 (index % 64))
    let replacement = Array.create 64 -1.0f
    use keys = upload backend [| 4; 64 |] first
    use replacementKeys = upload backend [| 1; 64 |] replacement
    use empty = upload backend [| 0; 64 |] [||]
    backend.StoreKeyValues(cache, 0, 0, keys, keys)

    // Regenerating from position 2 makes position 3, computed from the old token 2, unreadable.
    backend.StoreKeyValues(cache, 0, 2, replacementKeys, replacementKeys)
    Assert.Equal(3, cache.StoredPositions 0)
    let storedKeys, _ = read backend cache 0 0 3
    Assert.Equal<single>(Array.append first[.. 2 * 64 - 1] replacement, storedKeys)
    Assert.Throws<ArgumentOutOfRangeException>(fun () -> read backend cache 0 3 1 |> ignore) |> ignore

    // An empty batch at the end changes nothing; below the end, it only truncates the layer.
    backend.StoreKeyValues(cache, 0, 3, empty, empty)
    Assert.Equal(3, cache.StoredPositions 0)
    backend.StoreKeyValues(cache, 0, 1, empty, empty)
    Assert.Equal(1, cache.StoredPositions 0)

[<Fact>]
let ``key-value cache operations validate every argument before writing`` () =
    use backend = cpu ()
    use otherBackend = cpu ()
    use cache = backend.AllocateKeyValueCache(layout TensorEncoding.Q8_0 TensorEncoding.Q8_0)
    use foreignCache = otherBackend.AllocateKeyValueCache(layout TensorEncoding.Q8_0 TensorEncoding.Q8_0)
    use keys = upload backend [| 2; 64 |] (Array.create 128 1.0f)
    use values = upload backend [| 2; 64 |] (Array.create 128 1.0f)
    use oneToken = upload backend [| 1; 64 |] (Array.create 64 1.0f)
    use narrow = backend.Allocate(TensorShape.Create [| 1; 63 |])
    use wrongHeads = backend.Allocate(TensorShape.Create [| 1; 1; 64 |])
    use vector = backend.Allocate(TensorShape.Create [| 64 |])
    use foreign = otherBackend.Allocate(TensorShape.Create [| 2; 64 |])
    let disposed = backend.Allocate(TensorShape.Create [| 2; 64 |])
    disposed.Dispose()
    backend.StoreKeyValues(cache, 0, 0, keys, values)
    backend.StoreKeyValues(cache, 0, 2, oneToken, oneToken)

    let store layer position (keys: ITensor) (values: ITensor) () = backend.StoreKeyValues(cache, layer, position, keys, values)
    for layer in [ -1; 2 ] do
        Assert.Throws<ArgumentOutOfRangeException>(store layer 0 keys values) |> ignore
    let gap = Assert.Throws<ArgumentOutOfRangeException>(store 0 4 oneToken oneToken)
    Assert.Contains("contiguous", gap.Message)
    let full = Assert.Throws<ArgumentOutOfRangeException>(store 0 3 keys values)
    Assert.Contains("capacity", full.Message)
    Assert.Throws<ArgumentOutOfRangeException>(store 1 -1 oneToken oneToken) |> ignore
    Assert.Throws<ArgumentException>(store 1 0 narrow narrow) |> ignore
    Assert.Throws<ArgumentException>(store 1 0 wrongHeads wrongHeads) |> ignore
    Assert.Throws<ArgumentException>(store 1 0 vector vector) |> ignore
    Assert.Throws<ArgumentException>(store 1 0 keys oneToken) |> ignore
    Assert.Throws<ArgumentException>(store 1 0 foreign values) |> ignore
    Assert.Throws<ObjectDisposedException>(store 1 0 keys disposed) |> ignore
    Assert.Throws<ArgumentNullException>(store 1 0 Unchecked.defaultof<ITensor> values) |> ignore
    Assert.Throws<ArgumentException>(fun () -> backend.StoreKeyValues(foreignCache, 0, 0, keys, values)) |> ignore
    Assert.Throws<ArgumentNullException>(fun () -> backend.StoreKeyValues(Unchecked.defaultof<IKeyValueCache>, 0, 0, keys, values)) |> ignore
    Assert.Equal(3, cache.StoredPositions 0)
    Assert.Equal(0, cache.StoredPositions 1)

    use keyDestination = upload backend [| 2; 64 |] (Array.create 128 42.0f)
    use valueDestination = upload backend [| 2; 64 |] (Array.create 128 42.0f)
    let readInto layer position (keys: ITensor) (values: ITensor) () = backend.ReadKeyValues(cache, layer, position, keys, values)
    Assert.Throws<ArgumentOutOfRangeException>(readInto 1 0 keyDestination valueDestination) |> ignore
    Assert.Throws<ArgumentOutOfRangeException>(readInto 0 2 keyDestination valueDestination) |> ignore
    Assert.Throws<ArgumentOutOfRangeException>(readInto 0 -1 keyDestination valueDestination) |> ignore
    Assert.Throws<ArgumentOutOfRangeException>(readInto 2 0 keyDestination valueDestination) |> ignore
    Assert.Throws<ArgumentException>(readInto 0 0 keyDestination keyDestination) |> ignore
    Assert.Throws<ArgumentException>(readInto 0 0 keyDestination oneToken) |> ignore
    Assert.Throws<ArgumentException>(readInto 0 0 keyDestination foreign) |> ignore
    Assert.Throws<ArgumentException>(fun () -> backend.ReadKeyValues(foreignCache, 0, 0, keyDestination, valueDestination)) |> ignore
    Assert.Equal<single>(Array.create 128 42.0f, download backend keyDestination)
    Assert.Equal<single>(Array.create 128 42.0f, download backend valueDestination)

[<Fact>]
let ``key-value caches follow ownership and disposal rules`` () =
    let backend = cpu ()
    let cacheLayout = layout TensorEncoding.Q4_0 TensorEncoding.BF16
    let cache = backend.AllocateKeyValueCache(cacheLayout)
    let released = backend.AllocateKeyValueCache(cacheLayout)
    use keys = backend.Allocate(TensorShape.Create [| 1; 64 |])

    Assert.Same(cacheLayout, cache.Layout)
    Assert.Equal(BackendKind.Cpu, cache.Backend)
    Assert.Throws<ArgumentNullException>(fun () -> backend.AllocateKeyValueCache(Unchecked.defaultof<KeyValueCacheLayout>) |> ignore) |> ignore

    cache.Dispose()
    cache.Dispose()
    Assert.Same(cacheLayout, cache.Layout)
    Assert.Throws<ObjectDisposedException>(fun () -> cache.StoredPositions 0 |> ignore) |> ignore
    Assert.Throws<ObjectDisposedException>(fun () -> backend.StoreKeyValues(cache, 0, 0, keys, keys)) |> ignore

    // Disposing the backend releases the caches it still owns.
    backend.Dispose()
    Assert.Throws<ObjectDisposedException>(fun () -> released.StoredPositions 0 |> ignore) |> ignore
    released.Dispose()
    Assert.Throws<ObjectDisposedException>(fun () -> backend.AllocateKeyValueCache(cacheLayout) |> ignore) |> ignore
