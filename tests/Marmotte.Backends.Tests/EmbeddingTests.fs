module Marmotte.Backends.EmbeddingTests

open System
open System.Collections.Generic
open Marmotte.Core
open Marmotte.Cpu
open Xunit

/// Fixtures specify bytes independently of Marmotte's quantizer and decoder.
type private EmbeddingSource(encoding: TensorEncoding, rows: int, columns: int, payload: byte array) =
    let descriptor =
        { Name = "token_embd.weight"
          Encoding = Some encoding
          SourceEncoding = string encoding
          Shape = [| uint64 rows; uint64 columns |]
          ElementCount = uint64 rows * uint64 columns
          ByteSize = Some (uint64 payload.Length) }

    interface ITensorSource with
        member _.Tensors = [| descriptor |] :> IReadOnlyList<TensorDescriptor>
        member _.TryFindTensor(name) = if name = descriptor.Name then Some descriptor else None
        member _.ReadPayload(_) = ReadOnlyMemory<byte>(payload)

    interface IDisposable with
        member _.Dispose() = ()

let private uploadWeights (backend: ITensorBackend) (source: ITensorSource) =
    backend.UploadWeights(WeightMatrix.Read(source, "token_embd.weight"))

let private download (backend: ITensorBackend) (tensor: ITensor) =
    let result = Array.zeroCreate<single> tensor.Shape.ElementCount
    backend.Download(tensor, result.AsMemory())
    result

let private assertLookup encoding columns (payload: byte array) (expectedRows: single array array) =
    let original = Array.copy payload
    use source = new EmbeddingSource(encoding, expectedRows.Length, columns, payload)
    use backend = new CpuBackend() :> ITensorBackend
    use weights = uploadWeights backend source
    // Memory slices, last/first rows, repeated IDs and caller buffer reuse are all observable.
    let ids = [| -99; 1; 0; 1; -99 |]
    let shape = TensorShape.Create [| 3; columns |]
    use destination = backend.Allocate(shape)
    backend.EmbeddingLookup(weights, ReadOnlyMemory<int>(ids, 1, 3), destination)
    Array.fill ids 0 ids.Length 0
    let expected = Array.concat [ expectedRows[1]; expectedRows[0]; expectedRows[1] ]
    Assert.Equal<single>(expected, download backend destination)
    Assert.Equal<byte>(original, payload)

    // The output is an ordinary execution tensor and can be consumed by the next kernel.
    backend.Add(destination, destination, destination)
    Assert.Equal<single>(Array.map (fun value -> value + value) expected, download backend destination)

    // A single-token generation step still has a two-dimensional output.
    use singleToken = backend.Allocate(TensorShape.Create [| 1; columns |])
    backend.EmbeddingLookup(weights, ReadOnlyMemory<int>([| 0 |]), singleToken)
    Assert.Equal<single>(expectedRows[0], download backend singleToken)

let private floatingFixture = function
    | "F32" ->
        TensorEncoding.F32,
        Convert.FromHexString("0000803F000020C00000003E000080400000003F000040C0")
    | "F16" ->
        TensorEncoding.F16, Convert.FromHexString("003C00C100300044003800C2")
    | "BF16" ->
        TensorEncoding.BF16, Convert.FromHexString("803F20C0003E8040003F40C0")
    | _ -> invalidArg "encoding" "Unknown fixture encoding."

[<Theory>]
[<InlineData("F32")>]
[<InlineData("F16")>]
[<InlineData("BF16")>]
let ``lookup decodes little-endian floating weights in token order`` name =
    let encoding, payload = floatingFixture name
    assertLookup encoding 3 payload [| [| 1.0f; -2.5f; 0.125f |]; [| 4.0f; 0.5f; -3.0f |] |]

[<Theory>]
[<InlineData("F32", "00000080010000000000807F000080FF0000C07F", 1.401298464324817e-45f)>]
[<InlineData("F16", "00800100007C00FC007E", 5.960464477539063e-8f)>]
[<InlineData("BF16", "00800100807F80FFC07F", 9.183549615799121e-41f)>]
let ``floating lookup preserves signed zero subnormals infinities and NaN`` name (hex: string) (smallest: single) =
    let encoding, _ = floatingFixture name
    use source = new EmbeddingSource(encoding, 1, 5, Convert.FromHexString(hex))
    use backend = new CpuBackend() :> ITensorBackend
    use weights = uploadWeights backend source
    use destination = backend.Allocate(TensorShape.Create [| 1; 5 |])
    backend.EmbeddingLookup(weights, ReadOnlyMemory<int>([| 0 |]), destination)
    let actual = download backend destination
    Assert.Equal(Int32.MinValue, BitConverter.SingleToInt32Bits(actual[0]))
    Assert.Equal(smallest, actual[1])
    Assert.Equal(Single.PositiveInfinity, actual[2])
    Assert.Equal(Single.NegativeInfinity, actual[3])
    Assert.True(Single.IsNaN actual[4])

let private assertTwoBlockRows encoding (firstBlock: byte array) (secondBlock: byte array)
                               (firstValues: single array) (secondValues: single array) =
    let payload = Array.concat [ firstBlock; secondBlock; secondBlock; firstBlock ]
    let expected = [| Array.append firstValues secondValues; Array.append secondValues firstValues |]
    assertLookup encoding (2 * firstValues.Length) payload expected

[<Fact>]
let ``Q4_0 lookup orders nibbles and applies signed block scales across rows`` () =
    // Low nibbles encode -8..7; high nibbles encode 7..-8.
    let quants = Convert.FromHexString("F0E1D2C3B4A5968778695A4B3C2D1E0F")
    let values = Array.append [| -8.0f .. 7.0f |] [| 7.0f .. -1.0f .. -8.0f |]
    let first = Array.append [| 0x00uy; 0x38uy |] quants // F16 0.5
    let second = Array.append [| 0x00uy; 0xC0uy |] quants // F16 -2
    assertTwoBlockRows TensorEncoding.Q4_0 first second (Array.map ((*) 0.5f) values) (Array.map ((*) -2.0f) values)

[<Fact>]
let ``Q8_0 lookup treats quants as signed bytes including both endpoints`` () =
    let quants = Array.replicate 4 (Convert.FromHexString("80FF00017FC04020")) |> Array.concat
    let values = Array.replicate 4 [| -128.0f; -1.0f; 0.0f; 1.0f; 127.0f; -64.0f; 64.0f; 32.0f |] |> Array.concat
    let first = Array.append [| 0x00uy; 0x34uy |] quants // F16 0.25
    let second = Array.append [| 0x00uy; 0xC0uy |] quants // F16 -2
    assertTwoBlockRows TensorEncoding.Q8_0 first second (Array.map ((*) 0.25f) values) (Array.map ((*) -2.0f) values)

[<Fact>]
let ``Q4_K lookup unpacks all scale and minimum fields and both nibble planes`` () =
    // Six-bit factors, hand packed: includes high bits in each field and a zero scale.
    let factors = Convert.FromHexString("4192E334C5864738A9BCDEF0")
    let scales = [| 1.0f; 18.0f; 35.0f; 52.0f; 25.0f; 44.0f; 62.0f; 0.0f |]
    let minima = [| 5.0f; 6.0f; 7.0f; 56.0f; 58.0f; 43.0f; 29.0f; 15.0f |]
    let quants = Array.replicate 8 (Convert.FromHexString("F0E1D2C3B4A5968778695A4B3C2D1E0F")) |> Array.concat
    let first = Array.concat [ [| 0x00uy; 0x38uy; 0x00uy; 0x34uy |]; factors; quants ] // d=0.5, min=0.25
    let second = Array.concat [ [| 0x00uy; 0x40uy; 0x00uy; 0x3Cuy |]; factors; quants ] // d=2, min=1
    let expected scale minimum =
        [| for group = 0 to 7 do
               let values = if group % 2 = 0 then [| 0.0f .. 15.0f |] else [| 15.0f .. -1.0f .. 0.0f |]
               for quant in Array.append values values do
                   yield scale * scales[group] * quant - minimum * minima[group] |]
    assertTwoBlockRows TensorEncoding.Q4_K first second (expected 0.5f 0.25f) (expected 2.0f 1.0f)

[<Fact>]
let ``Q6_K lookup combines bit planes and signed scales for all sixteen groups`` () =
    let lowBits = Array.replicate 8 (Convert.FromHexString("F0E1D2C3B4A5968778695A4B3C2D1E0F")) |> Array.concat
    let highBits = Array.append (Array.create 32 0xE4uy) (Array.create 32 0x1Buy)
    let scaleBytes = Convert.FromHexString("80F9FDFF0001020304070B111F3F647F")
    let scales = [| -128.0f; -7.0f; -3.0f; -1.0f; 0.0f; 1.0f; 2.0f; 3.0f; 4.0f; 7.0f; 11.0f; 17.0f; 31.0f; 63.0f; 100.0f; 127.0f |]
    let first = Array.concat [ lowBits; highBits; scaleBytes; [| 0x00uy; 0x38uy |] ]
    let second = Array.concat [ lowBits; highBits; scaleBytes; [| 0x00uy; 0xC0uy |] ]
    let planes =
        [| [| -32.0f .. -17.0f |]; [| -16.0f .. -1.0f |]
           [| 15.0f .. -1.0f .. 0.0f |]; [| 31.0f .. -1.0f .. 16.0f |]
           [| 16.0f .. 31.0f |]; [| 0.0f .. 15.0f |]
           [| -1.0f .. -1.0f .. -16.0f |]; [| -17.0f .. -1.0f .. -32.0f |] |]
    let quants = planes |> Array.collect (fun plane -> Array.append plane plane)
    let expected scale = Array.mapi (fun index quant -> scale * scales[index / 16] * quant) quants
    assertTwoBlockRows TensorEncoding.Q6_K first second (expected 0.5f) (expected -2.0f)

[<Fact>]
let ``Q6_K high bits address the first and last lane of each half independently`` () =
    let block = Array.zeroCreate<byte> 210
    Array.fill block 192 16 1uy
    block[209] <- 0x3Cuy // d=1
    block[128] <- 0xE4uy
    block[159] <- 0x1Buy
    block[160] <- 0x39uy
    block[191] <- 0x93uy
    let expected = Array.create 256 -32.0f
    for index, value in
        [ 0, -32.0f; 32, -16.0f; 64, 0.0f; 96, 16.0f
          31, 16.0f; 63, 0.0f; 95, -16.0f; 127, -32.0f
          128, -16.0f; 160, 0.0f; 192, 16.0f; 224, -32.0f
          159, 16.0f; 191, -32.0f; 223, -16.0f; 255, 0.0f ] do
        expected[index] <- value
    assertLookup TensorEncoding.Q6_K 256 (Array.append block block) [| expected; expected |]

[<Fact>]
let ``Q6_K low nibbles address each plane and half independently`` () =
    let block = Array.zeroCreate<byte> 210
    Array.fill block 128 64 0xAAuy // High bits = 2; zero low nibbles reconstruct zero.
    Array.fill block 192 16 1uy
    block[209] <- 0x3Cuy
    for offset, packed in
        [ 0, 0xF1uy; 31, 0x2Euy; 32, 0xD3uy; 63, 0x4Cuy
          64, 0xB5uy; 95, 0x6Auy; 96, 0x97uy; 127, 0x88uy ] do
        block[offset] <- packed
    let expected = Array.zeroCreate<single> 256
    for index, value in
        [ 0, 1.0f; 64, 15.0f; 31, 14.0f; 95, 2.0f
          32, 3.0f; 96, 13.0f; 63, 12.0f; 127, 4.0f
          128, 5.0f; 192, 11.0f; 159, 10.0f; 223, 6.0f
          160, 7.0f; 224, 9.0f; 191, 8.0f; 255, 8.0f ] do
        expected[index] <- value
    assertLookup TensorEncoding.Q6_K 256 (Array.append block block) [| expected; expected |]

[<Fact>]
let ``Q4_K quants address each pair of groups independently`` () =
    let block = Array.zeroCreate<byte> 144
    block[1] <- 0x3Cuy
    Array.fill block 4 4 1uy
    Array.fill block 12 4 1uy // All eight scales = 1, all minima = 0.
    for offset, packed in [ 16, 0xF1uy; 47, 0x2Euy; 48, 0xD3uy; 79, 0x4Cuy; 80, 0xB5uy; 111, 0x6Auy; 112, 0x97uy; 143, 0x88uy ] do
        block[offset] <- packed
    let expected = Array.zeroCreate<single> 256
    for index, value in
        [ 0, 1.0f; 32, 15.0f; 31, 14.0f; 63, 2.0f
          64, 3.0f; 96, 13.0f; 95, 12.0f; 127, 4.0f
          128, 5.0f; 160, 11.0f; 159, 10.0f; 191, 6.0f
          192, 7.0f; 224, 9.0f; 223, 8.0f; 255, 8.0f ] do
        expected[index] <- value
    assertLookup TensorEncoding.Q4_K 256 (Array.append block block) [| expected; expected |]

[<Theory>]
[<InlineData(-1)>]
[<InlineData(2)>]
[<InlineData(Int32.MinValue)>]
[<InlineData(Int32.MaxValue)>]
let ``invalid IDs leave the complete destination unchanged`` invalidId =
    let encoding, payload = floatingFixture "F32"
    use source = new EmbeddingSource(encoding, 2, 3, payload)
    use backend = new CpuBackend() :> ITensorBackend
    use weights = uploadWeights backend source
    let sentinel = Array.create 9 42.0f
    use destination = backend.Upload(TensorShape.Create [| 3; 3 |], ReadOnlyMemory<single>(sentinel))
    let error = Assert.Throws<ArgumentOutOfRangeException>(fun () ->
        backend.EmbeddingLookup(weights, ReadOnlyMemory<int>([| 1; 0; invalidId |]), destination))
    Assert.Equal("tokenIds", error.ParamName)
    Assert.Contains("index 2", error.Message)
    Assert.Equal<single>(sentinel, download backend destination)

[<Fact>]
let ``lookup requires exact rank and dimensions and accepts an empty batch`` () =
    let encoding, payload = floatingFixture "F32"
    use source = new EmbeddingSource(encoding, 2, 3, payload)
    use backend = new CpuBackend() :> ITensorBackend
    use weights = uploadWeights backend source
    for dimensions in [ [| 6 |]; [| 3; 2 |]; [| 1; 2; 3 |]; [| 1; 3 |]; [||] ] do
        let shape = TensorShape.Create dimensions
        let sentinel = Array.create shape.ElementCount 42.0f
        use destination = backend.Upload(shape, ReadOnlyMemory<single>(sentinel))
        Assert.Throws<ArgumentException>(fun () ->
            backend.EmbeddingLookup(weights, ReadOnlyMemory<int>([| 0; 1 |]), destination)) |> ignore
        Assert.Equal<single>(sentinel, download backend destination)

    use empty = backend.Allocate(TensorShape.Create [| 0; 3 |])
    backend.EmbeddingLookup(weights, ReadOnlyMemory<int>.Empty, empty)
    Assert.Empty(download backend empty)
    use wrongEmpty = backend.Allocate(TensorShape.Create [| 0; 4 |])
    Assert.Throws<ArgumentException>(fun () ->
        backend.EmbeddingLookup(weights, ReadOnlyMemory<int>.Empty, wrongEmpty)) |> ignore

[<Fact>]
let ``lookup rejects null foreign and disposed tensors before writing`` () =
    let encoding, payload = floatingFixture "F32"
    use source = new EmbeddingSource(encoding, 2, 3, payload)
    use backend = new CpuBackend() :> ITensorBackend
    use other = new CpuBackend() :> ITensorBackend
    use weights = uploadWeights backend source
    use foreignWeights = uploadWeights other source
    let shape = TensorShape.Create [| 1; 3 |]
    let sentinel = [| 42.0f; 42.0f; 42.0f |]
    use destination = backend.Upload(shape, ReadOnlyMemory<single>(sentinel))
    use foreignDestination = other.Upload(shape, ReadOnlyMemory<single>(sentinel))
    use disposedWeights = uploadWeights backend source
    use disposedDestination = backend.Allocate(shape)
    disposedWeights.Dispose()
    disposedWeights.Dispose()
    disposedDestination.Dispose()
    let ids = ReadOnlyMemory<int>([| 1 |])

    Assert.Throws<ArgumentNullException>(fun () -> backend.EmbeddingLookup(Unchecked.defaultof<IWeightTensor>, ids, destination)) |> ignore
    Assert.Throws<ArgumentNullException>(fun () -> backend.EmbeddingLookup(weights, ids, Unchecked.defaultof<ITensor>)) |> ignore
    Assert.Throws<ArgumentException>(fun () -> backend.EmbeddingLookup(foreignWeights, ids, destination)) |> ignore
    Assert.Throws<ArgumentException>(fun () -> backend.EmbeddingLookup(weights, ids, foreignDestination)) |> ignore
    Assert.Throws<ObjectDisposedException>(fun () -> backend.EmbeddingLookup(disposedWeights, ids, destination)) |> ignore
    Assert.Throws<ObjectDisposedException>(fun () -> backend.EmbeddingLookup(weights, ids, disposedDestination)) |> ignore
    Assert.Equal<single>(sentinel, download backend destination)
    Assert.Equal<single>(sentinel, download other foreignDestination)
