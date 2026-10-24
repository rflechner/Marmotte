module Marmotte.Gguf.QuantizationTests

open System
open System.IO
open System.Text
open System.Threading
open Marmotte.Gguf
open Xunit

// Independent wire fixtures: no production writer or codec is used to construct input.
let private number big width (value: uint64) =
    let bytes = Array.init width (fun index -> byte (value >>> (index * 8)))
    if big then Array.rev bytes else bytes

let private text big (value: string) =
    let bytes = Encoding.UTF8.GetBytes value
    Array.append (number big 8 (uint64 bytes.Length)) bytes

let private metadata big key tag bytes =
    Array.concat [ text big key; number big 4 tag; bytes ]

type private Tensor =
    { Name: string
      Dimensions: uint64 list
      Tag: uint64
      Bytes: byte array }

let private floats tag (values: single array) =
    values |> Array.collect (fun value ->
        match tag with
        | 0UL -> number false 4 (uint64 (BitConverter.SingleToUInt32Bits value))
        | 1UL -> number false 2 (uint64 (BitConverter.HalfToUInt16Bits(Half.op_Explicit value)))
        | 30UL -> number false 2 (uint64 (BitConverter.SingleToUInt32Bits value >>> 16))
        | _ -> failwith "Invalid test encoding")

let private matrix tag values =
    { Name = "layer.weight"; Dimensions = [ 32UL; uint64 (Array.length values / 32) ]
      Tag = tag; Bytes = floats tag values }

let private fixture big version alignment extraMetadata (tensors: Tensor list) =
    let align position = position + (alignment - position % alignment) % alignment
    let mutable offset = 0
    let descriptors = tensors |> List.map (fun tensor ->
        offset <- align offset
        let descriptor =
            Array.concat [ text big tensor.Name; number big 4 (uint64 tensor.Dimensions.Length)
                           tensor.Dimensions |> List.map (number big 8) |> Array.concat
                           number big 4 tensor.Tag; number big 8 (uint64 offset) ]
        offset <- offset + tensor.Bytes.Length
        descriptor)
    let entries = metadata big "general.alignment" 4UL (number big 4 (uint64 alignment)) :: extraMetadata
    let index =
        Array.concat [ Encoding.ASCII.GetBytes "GGUF"; number big 4 version
                       number big 8 (uint64 tensors.Length); number big 8 (uint64 entries.Length)
                       Array.concat entries; Array.concat descriptors ]
    let dataOffset = align index.Length
    let bytes = Array.zeroCreate<byte> (dataOffset + offset)
    Array.Copy(index, bytes, index.Length)
    let mutable position = 0
    for tensor in tensors do
        position <- align position
        Array.Copy(tensor.Bytes, 0, bytes, dataOffset + position, tensor.Bytes.Length)
        position <- position + tensor.Bytes.Length
    bytes

type private Files(bytes: byte array) =
    let directory = Directory.CreateTempSubdirectory("marmotte-quantization-tests-")
    let input = Path.Combine(directory.FullName, "input.gguf")
    let output = Path.Combine(directory.FullName, "output.gguf")
    do File.WriteAllBytes(input, bytes)
    member _.Input = input
    member _.Output = output
    member _.Directory = directory.FullName
    interface IDisposable with
        member _.Dispose() = directory.Delete(true)

let private quantize format (files: Files) = GgufQuantizer.quantizeFile format files.Input files.Output

let private payload path name =
    let tensor = (Gguf.readFile path).Tensors |> Array.find (fun tensor -> tensor.Name = name)
    use stream = File.OpenRead path
    stream.Position <- tensor.AbsoluteOffset
    let bytes = Array.zeroCreate<byte> (int tensor.ByteSize.Value)
    stream.ReadExactly(bytes)
    bytes

let private decode format (bytes: byte array) =
    let blockSize = if format = GgufQuantizationFormat.Q4_0 then 18 else 34
    Array.init (bytes.Length / blockSize * 32) (fun index ->
        let start = index / 32 * blockSize
        let element = index % 32
        let scaleBits = uint16 bytes[start] ||| (uint16 bytes[start + 1] <<< 8)
        let scale = float32 (BitConverter.UInt16BitsToHalf scaleBits)
        let quantized =
            if format = GgufQuantizationFormat.Q4_0 then
                let packed = int bytes[start + 2 + element % 16]
                (if element < 16 then packed &&& 15 else packed >>> 4) - 8
            else
                let raw = int bytes[start + 2 + element]
                if raw > 127 then raw - 256 else raw
        scale * float32 quantized)

[<Theory>]
[<InlineData(0UL, false)>]
[<InlineData(1UL, false)>]
[<InlineData(30UL, false)>]
[<InlineData(0UL, true)>]
[<InlineData(1UL, true)>]
[<InlineData(30UL, true)>]
let ``writes exact GGML blocks from all supported floating encodings`` tag q4 =
    let format, firstBlock, packed =
        if q4 then
            GgufQuantizationFormat.Q4_0,
            Array.init 32 (fun index -> float32 (index % 16 - 8)),
            Array.init 16 (fun index -> byte (index * 17))
        else
            GgufQuantizationFormat.Q8_0,
            Array.init 32 (fun index -> [| -127.f; -64.f; -1.f; 0.f; 1.f; 64.f; 127.f; 0.f |][index % 8]),
            Array.init 32 (fun index -> [| 129uy; 192uy; 255uy; 0uy; 1uy; 64uy; 127uy; 0uy |][index % 8])
    let values = Array.append firstBlock (firstBlock |> Array.map ((*) 2.0f))
    let input = fixture false 2UL 32 [] [ matrix tag values ]
    use files = new Files(input)
    let result = quantize format files
    let model = Gguf.readFile files.Output
    let expected = Array.concat [ [| 0uy; 60uy |]; packed; [| 0uy; 64uy |]; packed ]

    Assert.Equal<byte>(expected, payload files.Output "layer.weight")
    Assert.Equal<single>(values, decode format expected)
    Assert.Equal<byte>(input, File.ReadAllBytes files.Input)
    Assert.Equal(3u, model.Version)
    Assert.Equal(GgufEndianness.LittleEndian, model.Endianness)
    Assert.Equal(Some (if q4 then GgufFileType.Q4_0 else GgufFileType.Q8_0), model.FileType)
    Assert.Equal(GgufValue.UInt32 2u, model.Metadata["general.quantization_version"])
    Assert.Equal(1, result.QuantizedTensors)
    Assert.Equal(0, result.PreservedTensors)
    Assert.Equal(int64 input.Length, result.InputBytes)
    Assert.Equal(FileInfo(files.Output).Length, result.OutputBytes)

[<Theory>]
[<InlineData(false)>]
[<InlineData(true)>]
let ``bounds reconstruction error across multiple processing chunks`` q4 =
    let format = if q4 then GgufQuantizationFormat.Q4_0 else GgufQuantizationFormat.Q8_0
    let random = Random(42)
    let values = Array.init (32 * 1026) (fun _ -> float32 (random.NextDouble() * 6.0 - 3.0))
    use files = new Files(fixture false 3UL 32 [] [ matrix 0UL values ])
    let result = quantize format files
    let decoded = decode format (payload files.Output "layer.weight")
    Assert.Equal(values.Length, decoded.Length)
    for block in 0 .. 1025 do
        let maximum = values[block * 32 .. block * 32 + 31] |> Array.map abs |> Array.max
        // Q4's asymmetric [-8,7] range can clip one endpoint by one step.
        let roundingError = if q4 then maximum / 8.f else maximum / 254.f
        // The block scale itself is rounded to F16 (up to 2^-11 relative error).
        let tolerance = roundingError + maximum / 2048.f
        for index in block * 32 .. block * 32 + 31 do
            Assert.InRange(abs (values[index] - decoded[index]), 0.f, tolerance + 0.000001f)
    Assert.True(result.OutputBytes < result.InputBytes)

[<Fact>]
let ``uses GGML signed Q4 scale and Q8 rounding away from zero`` () =
    let values = Array.concat [ [| 127.f; 0.5f; -0.5f; 1.5f; -1.5f |]; Array.zeroCreate 27 ]
    use q8Files = new Files(fixture false 3UL 32 [] [ matrix 0UL values ])
    quantize GgufQuantizationFormat.Q8_0 q8Files |> ignore
    let bytes = payload q8Files.Output "layer.weight"
    Assert.Equal<byte>([| 0uy; 60uy; 127uy; 1uy; 255uy; 2uy; 254uy |], bytes[..6])

    use q4Files = new Files(fixture false 3UL 32 [] [ matrix 0UL (Array.create 32 8.f) ])
    quantize GgufQuantizationFormat.Q4_0 q4Files |> ignore
    Assert.Equal<byte>(Array.append [| 0uy; 188uy |] (Array.zeroCreate 16), payload q4Files.Output "layer.weight")

[<Theory>]
[<InlineData(false, 0.0f)>]
[<InlineData(true, 0.0f)>]
[<InlineData(false, Single.Epsilon)>]
[<InlineData(true, Single.Epsilon)>]
let ``zero and subnormal blocks decode to zero without nonfinite scales`` q4 value =
    let format = if q4 then GgufQuantizationFormat.Q4_0 else GgufQuantizationFormat.Q8_0
    use files = new Files(fixture false 3UL 32 [] [ matrix 0UL (Array.create 32 value) ])
    quantize format files |> ignore
    let decoded = decode format (payload files.Output "layer.weight")
    Assert.All(decoded, fun actual -> Assert.Equal(0.f, actual))

[<Fact>]
let ``preserves other tensors and all metadata types with nonpoweroftwo alignment`` () =
    let scalarEntries =
        [ 0UL, [| 255uy |]; 1UL, [| 128uy |]; 2UL, number false 2 65000UL
          3UL, number false 2 32768UL; 4UL, number false 4 123UL; 5UL, number false 4 0x80000000UL
          6UL, number false 4 0x3fc00000UL; 7UL, [| 1uy |]; 8UL, text false "Unicode λ 🦫"
          10UL, number false 8 UInt64.MaxValue; 11UL, number false 8 0x8000000000000000UL
          12UL, number false 8 0xc002000000000000UL ]
    let entries =
        [ for tag, bytes in scalarEntries do
              yield metadata false $"test.scalar_{tag}" tag bytes
              yield metadata false $"test.array_{tag}" 9UL (Array.concat [ number false 4 tag; number false 8 2UL; bytes; bytes ])
          yield metadata false "test.empty" 9UL (Array.concat [ number false 4 8UL; number false 8 0UL ])
          yield metadata false "test.nested" 9UL (Array.concat [ number false 4 9UL; number false 8 1UL
                                                                 number false 4 4UL; number false 8 1UL; number false 4 123UL ])
          yield metadata false "general.file_type" 4UL (number false 4 1UL)
          yield metadata false "general.quantization_version" 4UL (number false 4 1UL) ]
    let preserved =
        [ { Name = "norm.weight"; Dimensions = [ 32UL ]; Tag = 1UL; Bytes = floats 1UL (Array.create 32 1.f) }
          { Name = "odd.weight"; Dimensions = [ 3UL; 2UL ]; Tag = 30UL; Bytes = floats 30UL (Array.create 6 0.5f) }
          { Name = "scalar"; Dimensions = []; Tag = 0UL; Bytes = floats 0UL [| -2.25f |] }
          { Name = "empty"; Dimensions = [ 0UL; 32UL ]; Tag = 0UL; Bytes = [||] } ]
    let tensors = List.head preserved :: matrix 0UL (Array.create 64 1.f) :: List.tail preserved
    use files = new Files(fixture false 3UL 24 entries tensors)
    let before = Gguf.readFile files.Input
    let result = quantize GgufQuantizationFormat.Q4_0 files
    let after = Gguf.readFile files.Output
    Assert.Equal(1, result.QuantizedTensors)
    Assert.Equal(4, result.PreservedTensors)
    Assert.Equal(24u, after.Alignment)
    let retainedMetadata model = model.Metadata |> Map.remove "general.file_type" |> Map.remove "general.quantization_version"
    Assert.Equal<Map<string, GgufValue>>(retainedMetadata before, retainedMetadata after)
    for tensor in preserved do
        Assert.Equal<byte>(tensor.Bytes, payload files.Output tensor.Name)
        let descriptor = after.Tensors |> Array.find (fun item -> item.Name = tensor.Name)
        Assert.Equal(tensor.Tag, uint64 descriptor.TensorType)
    for index in 0 .. after.Tensors.Length - 1 do
        Assert.Equal<uint64>(before.Tensors[index].Dimensions, after.Tensors[index].Dimensions)
        Assert.Equal(0UL, after.Tensors[index].Offset % 24UL)
        Assert.Equal(0L, after.Tensors[index].AbsoluteOffset % 24L)

[<Theory>]
[<InlineData(2UL, 18)>]
[<InlineData(12UL, 144)>]
[<InlineData(24UL, 32)>]
[<InlineData(999UL, 32)>]
let ``rejects unsupported input encodings before creating output`` tag byteCount =
    let width = if tag = 12UL then 256UL else 32UL
    let tensor = { Name = "layer.weight"; Dimensions = [ width; 1UL ]; Tag = tag; Bytes = Array.zeroCreate byteCount }
    use files = new Files(fixture false 3UL 32 [] [ tensor ])
    let error = Assert.Throws<InvalidDataException>(fun () -> quantize GgufQuantizationFormat.Q8_0 files |> ignore)
    Assert.Contains("Only F32, F16 and BF16", error.Message)
    Assert.False(File.Exists files.Output)
    Assert.Single(Directory.GetFiles files.Directory) |> ignore

[<Theory>]
[<InlineData(true)>]
[<InlineData(false)>]
let ``rejects big endian and split input`` big =
    let entries = if big then [] else [ metadata false "split.count" 2UL (number false 2 2UL) ]
    use files = new Files(fixture big 3UL 32 entries [ matrix 0UL (Array.zeroCreate 32) ])
    let error = Assert.Throws<InvalidDataException>(fun () -> quantize GgufQuantizationFormat.Q8_0 files |> ignore)
    Assert.Contains((if big then "little-endian" else "Split GGUF"), error.Message)
    Assert.Single(Directory.GetFiles files.Directory) |> ignore

[<Fact>]
let ``rejects files with no eligible tensors`` () =
    let tensor = { Name = "vector"; Dimensions = [ 32UL ]; Tag = 0UL; Bytes = floats 0UL (Array.zeroCreate 32) }
    use files = new Files(fixture false 3UL 32 [] [ tensor ])
    let error = Assert.Throws<InvalidDataException>(fun () -> quantize GgufQuantizationFormat.Q8_0 files |> ignore)
    Assert.Contains("No eligible tensors", error.Message)
    Assert.Single(Directory.GetFiles files.Directory) |> ignore

[<Theory>]
[<InlineData(Single.NaN)>]
[<InlineData(Single.PositiveInfinity)>]
[<InlineData(Single.NegativeInfinity)>]
[<InlineData(Single.MaxValue)>]
let ``removes partial output when a later block contains invalid weights`` value =
    let values = Array.zeroCreate<single> (32 * 513)
    values[values.Length - 1] <- value
    let bytes = fixture false 3UL 32 [] [ matrix 0UL values ]
    use files = new Files(bytes)
    let error = Assert.Throws<InvalidDataException>(fun () -> quantize GgufQuantizationFormat.Q8_0 files |> ignore)
    Assert.Contains("layer.weight", error.Message)
    Assert.Single(Directory.GetFiles files.Directory) |> ignore
    Assert.Equal<byte>(bytes, File.ReadAllBytes files.Input)

[<Fact>]
let ``does not overwrite input existing output or output directories`` () =
    let bytes = fixture false 3UL 32 [] [ matrix 0UL (Array.zeroCreate 32) ]
    use files = new Files(bytes)
    Assert.Throws<ArgumentException>(fun () ->
        GgufQuantizer.quantizeFile GgufQuantizationFormat.Q8_0 files.Input files.Input |> ignore) |> ignore
    let existing = [| 1uy; 2uy; 3uy |]
    File.WriteAllBytes(files.Output, existing)
    Assert.Throws<IOException>(fun () -> quantize GgufQuantizationFormat.Q8_0 files |> ignore) |> ignore
    Assert.Throws<IOException>(fun () ->
        GgufQuantizer.quantizeFile GgufQuantizationFormat.Q8_0 files.Input files.Directory |> ignore) |> ignore
    Assert.Equal<byte>(bytes, File.ReadAllBytes files.Input)
    Assert.Equal<byte>(existing, File.ReadAllBytes files.Output)

[<Fact>]
let ``cancellation leaves input intact and creates no output`` () =
    let bytes = fixture false 3UL 32 [] [ matrix 0UL (Array.zeroCreate 32) ]
    use files = new Files(bytes)
    use cancellation = new CancellationTokenSource()
    cancellation.Cancel()
    Assert.Throws<OperationCanceledException>(fun () ->
        GgufQuantizer.quantizeFileWithCancellation GgufQuantizationFormat.Q8_0
            files.Input files.Output cancellation.Token |> ignore) |> ignore
    Assert.Single(Directory.GetFiles files.Directory) |> ignore
    Assert.Equal<byte>(bytes, File.ReadAllBytes files.Input)

[<Fact>]
let ``truncated weights are rejected without publishing output`` () =
    let bytes = fixture false 3UL 32 [] [ matrix 0UL (Array.zeroCreate 32) ]
    use files = new Files(bytes[..bytes.Length - 2])
    Assert.Throws<InvalidDataException>(fun () -> quantize GgufQuantizationFormat.Q8_0 files |> ignore) |> ignore
    Assert.Single(Directory.GetFiles files.Directory) |> ignore
