module Marmotte.Gguf.Tests

open System
open System.IO
open System.Text
open Marmotte.Gguf
open Xunit

// Fixtures encode the wire format directly, independently of the parser's type/layout tables.
let private number big width (value: uint64) =
    let bytes = Array.init width (fun index -> byte (value >>> (index * 8)))
    if big then Array.rev bytes else bytes

let private u32 big value = number big 4 (uint64 value)
let private u64 big value = number big 8 value
let private concat parts = Array.concat parts
let private str big (value: string) =
    let bytes = Encoding.UTF8.GetBytes(value)
    concat [ u64 big (uint64 bytes.Length); bytes ]

let private kv big key tag payload = concat [ str big key; u32 big tag; payload ]
let private arr big tag (values: byte array list) =
    concat [ u32 big tag; u64 big (uint64 values.Length); concat values ]

let private tensor big name dimensions tag offset =
    concat [ str big name; u32 big (uint32 (List.length dimensions))
             concat (List.map (u64 big) dimensions); u32 big tag; u64 big offset ]

let private header big version tensors metadata =
    concat [ [| 0x47uy; 0x47uy; 0x55uy; 0x46uy |]; u32 big version
             u64 big tensors; u64 big metadata ]

let private file big version alignment metadata tensors payload =
    let index = concat [ header big version (uint64 (List.length tensors)) (uint64 (List.length metadata))
                         concat metadata; concat tensors ]
    let padding = (alignment - index.Length % alignment) % alignment
    concat [ index; Array.zeroCreate padding; payload ]

let private parse bytes =
    use stream = new MemoryStream(bytes, false)
    Gguf.read stream

let private parseWith options bytes =
    use stream = new MemoryStream(bytes, false)
    Gguf.readWithOptions options stream

let private invalid bytes =
    let error = Assert.Throws<InvalidDataException>(fun () -> parse bytes |> ignore)
    Assert.Contains("byte", error.Message)
    error

let private scalarCases big =
    [ 0u, [| 255uy |], GgufValue.UInt8 255uy
      1u, [| 0x80uy |], GgufValue.Int8 -128y
      2u, number big 2 65000UL, GgufValue.UInt16 65000us
      3u, number big 2 0x8000UL, GgufValue.Int16 -32768s
      4u, u32 big UInt32.MaxValue, GgufValue.UInt32 UInt32.MaxValue
      5u, u32 big 0x80000000u, GgufValue.Int32 Int32.MinValue
      6u, u32 big 0x3fc00000u, GgufValue.Float32 1.5f
      7u, [| 1uy |], GgufValue.Bool true
      8u, str big "Unicode λ 🦫\000", GgufValue.String "Unicode λ 🦫\000"
      10u, u64 big UInt64.MaxValue, GgufValue.UInt64 UInt64.MaxValue
      11u, u64 big 0x8000000000000000UL, GgufValue.Int64 Int64.MinValue
      12u, u64 big 0xc002000000000000UL, GgufValue.Float64 -2.25 ]

[<Theory>]
[<InlineData(false, 2u)>]
[<InlineData(false, 3u)>]
[<InlineData(true, 3u)>]
let ``reads all scalar types and typed arrays`` big version =
    let cases = scalarCases big
    let metadata =
        cases |> List.collect (fun (tag, payload, _) ->
            [ kv big $"test.scalar_{tag}" tag payload
              kv big $"test.array_{tag}" 9u (arr big tag [ payload; payload ])
              kv big $"test.empty_{tag}" 9u (arr big tag []) ])
    let model = parse (file big version 32 metadata [] [||])
    Assert.Equal(version, model.Version)
    Assert.Equal((if big then GgufEndianness.BigEndian else GgufEndianness.LittleEndian), model.Endianness)
    Assert.Equal(32u, model.Alignment)
    Assert.Empty(model.Tensors)
    for tag, _, expected in cases do
        Assert.Equal(expected, model.Metadata[$"test.scalar_{tag}"])
        let elementType = LanguagePrimitives.EnumOfValue<uint32, GgufValueType> tag
        Assert.Equal(GgufValue.Array(elementType, [| expected; expected |]), model.Metadata[$"test.array_{tag}"])
        Assert.Equal(GgufValue.Array(elementType, [||]), model.Metadata[$"test.empty_{tag}"])

[<Theory>]
[<InlineData(false)>]
[<InlineData(true)>]
let ``reads nested arrays and empty strings`` big =
    let metadata =
        [ kv big "test.nested" 9u (arr big 9u [ arr big 8u [ str big "" ]; arr big 7u [ [| 0uy |] ] ])
          kv big "test.empty" 9u (arr big 9u []) ]
    let model = parse (file big 3u 32 metadata [] [||])
    let expected = GgufValue.Array(GgufValueType.Array, [|
        GgufValue.Array(GgufValueType.String, [| GgufValue.String "" |])
        GgufValue.Array(GgufValueType.Bool, [| GgufValue.Bool false |]) |])
    Assert.Equal(expected, model.Metadata["test.nested"])
    Assert.Equal(GgufValue.Array(GgufValueType.Array, [||]), model.Metadata["test.empty"])

[<Theory>]
[<InlineData(false, 8)>]
[<InlineData(false, 24)>]
[<InlineData(false, 64)>]
[<InlineData(true, 64)>]
let ``resolves tensor offsets with custom alignment and preserves descriptor order`` big alignment =
    let metadata = [ kv big "general.alignment" 4u (u32 big (uint32 alignment)) ]
    let descriptors =
        [ tensor big "quantized.weight" [ 32UL; 2UL ] 2u (uint64 alignment)
          tensor big "bias" [ 2UL ] 0u 0UL ]
    let bytes = file big 3u alignment metadata descriptors (Array.zeroCreate (alignment + 36))
    let model = parse bytes
    Assert.Equal(0L, model.DataOffset % int64 alignment)
    Assert.Equal("quantized.weight", model.Tensors[0].Name)
    Assert.Equal<uint64>([| 32UL; 2UL |], model.Tensors[0].Dimensions)
    Assert.Equal(GgmlType.Q4_0, model.Tensors[0].TensorType)
    Assert.Equal(64UL, model.Tensors[0].ElementCount)
    Assert.Equal(Some 36UL, model.Tensors[0].ByteSize)
    Assert.Equal(model.DataOffset + int64 alignment, model.Tensors[0].AbsoluteOffset)
    Assert.Equal(Some 8UL, model.Tensors[1].ByteSize)
    Assert.Equal(model.DataOffset, model.Tensors[1].AbsoluteOffset)

[<Theory>]
[<InlineData(0u, 1UL, 4)>]
[<InlineData(1u, 1UL, 2)>]
[<InlineData(2u, 32UL, 18)>]
[<InlineData(3u, 32UL, 20)>]
[<InlineData(6u, 32UL, 22)>]
[<InlineData(7u, 32UL, 24)>]
[<InlineData(8u, 32UL, 34)>]
[<InlineData(9u, 32UL, 36)>]
[<InlineData(10u, 256UL, 84)>]
[<InlineData(11u, 256UL, 110)>]
[<InlineData(12u, 256UL, 144)>]
[<InlineData(13u, 256UL, 176)>]
[<InlineData(14u, 256UL, 210)>]
[<InlineData(15u, 256UL, 292)>]
[<InlineData(16u, 256UL, 66)>]
[<InlineData(17u, 256UL, 74)>]
[<InlineData(18u, 256UL, 98)>]
[<InlineData(19u, 256UL, 50)>]
[<InlineData(20u, 32UL, 18)>]
[<InlineData(21u, 256UL, 110)>]
[<InlineData(22u, 256UL, 82)>]
[<InlineData(23u, 256UL, 136)>]
[<InlineData(24u, 1UL, 1)>]
[<InlineData(25u, 1UL, 2)>]
[<InlineData(26u, 1UL, 4)>]
[<InlineData(27u, 1UL, 8)>]
[<InlineData(28u, 1UL, 8)>]
[<InlineData(29u, 256UL, 56)>]
[<InlineData(30u, 1UL, 2)>]
[<InlineData(34u, 256UL, 54)>]
[<InlineData(35u, 256UL, 66)>]
[<InlineData(39u, 32UL, 17)>]
[<InlineData(40u, 64UL, 36)>]
[<InlineData(41u, 128UL, 18)>]
[<InlineData(42u, 64UL, 18)>]
let ``validates payload sizes for supported encodings`` tag block bytesPerBlock =
    let bytes = file false 3u 32 [] [ tensor false "weight" [ block; 2UL ] tag 0UL ] (Array.zeroCreate (bytesPerBlock * 2))
    let model = parse bytes
    Assert.Equal(Some(uint64 (bytesPerBlock * 2)), model.Tensors[0].ByteSize)
    invalid bytes[..bytes.Length - 2] |> ignore

[<Fact>]
let ``preserves unknown tensor encoding identifiers`` () =
    let bytes = file false 3u 32 [] [ tensor false "future" [ 3UL ] 123456u 0UL ] [| 42uy |]
    let info = (parse bytes).Tensors[0]
    Assert.Equal(123456u, uint32 info.TensorType)
    Assert.Equal(None, info.ByteSize)

[<Theory>]
[<InlineData(false, 11u, GgufFileType.Q3_K_S)>]
[<InlineData(false, 12u, GgufFileType.Q3_K_M)>]
[<InlineData(false, 13u, GgufFileType.Q3_K_L)>]
[<InlineData(true, 11u, GgufFileType.Q3_K_S)>]
[<InlineData(true, 12u, GgufFileType.Q3_K_M)>]
[<InlineData(true, 13u, GgufFileType.Q3_K_L)>]
let ``file profile is independent of the actual tensor encodings`` big profile expected =
    let metadata = [ kv big "general.file_type" 4u (u32 big profile) ]
    let tensors = [ tensor big "weight" [ 256UL ] 11u 0UL
                    tensor big "output.weight" [ 256UL ] 14u 128UL
                    tensor big "bias" [ 1UL ] 0u 352UL ]
    let bytes = file big 3u 32 metadata tensors (Array.zeroCreate 356)
    let model = parse bytes
    Assert.Equal(Some expected, model.FileType)
    Assert.Equal(GgufValue.UInt32 profile, model.Metadata["general.file_type"])
    Assert.Equal(GgmlType.Q3_K, model.Tensors[0].TensorType)
    Assert.Equal(Some 110UL, model.Tensors[0].ByteSize)
    Assert.Equal(GgmlType.Q6_K, model.Tensors[1].TensorType)
    Assert.Equal(Some 210UL, model.Tensors[1].ByteSize)
    Assert.Equal(GgmlType.F32, model.Tensors[2].TensorType)
    Assert.Equal(Some 4UL, model.Tensors[2].ByteSize)

[<Fact>]
let ``does not guess a missing file profile from tensor types`` () =
    let bytes = file false 3u 32 [] [ tensor false "weight" [ 256UL ] 11u 0UL ] (Array.zeroCreate 110)
    Assert.Equal(None, (parse bytes).FileType)

[<Fact>]
let ``preserves unknown file profile identifiers`` () =
    let metadata = [ kv false "general.file_type" 4u (u32 false 123456u) ]
    let model = parse (file false 3u 32 metadata [] [||])
    Assert.Equal(Some 123456u, model.FileType |> Option.map uint32)

[<Fact>]
let ``preserves metadata when file profile has an unexpected value type`` () =
    let metadata = [ kv false "general.file_type" 8u (str false "Q3_K_S") ]
    let model = parse (file false 3u 32 metadata [] [||])
    Assert.Equal(None, model.FileType)
    Assert.Equal(GgufValue.String "Q3_K_S", model.Metadata["general.file_type"])

[<Fact>]
let ``reads scalar and empty tensors`` () =
    let scalar = file false 3u 32 [] [ tensor false "scalar" [] 0u 0UL ] (Array.zeroCreate 4)
    let info = (parse scalar).Tensors[0]
    Assert.Empty(info.Dimensions)
    Assert.Equal(1UL, info.ElementCount)
    Assert.Equal(Some 4UL, info.ByteSize)
    let descriptors = [ tensor false "empty" [ 0UL ] 0u 0UL
                        tensor false "empty_quantized" [ 0UL; 32UL ] 2u 0UL ]
    let model = parse (file false 3u 32 [] descriptors [||])
    for tensor in model.Tensors do
        Assert.Equal(0UL, tensor.ElementCount)
        Assert.Equal(Some 0UL, tensor.ByteSize)
        Assert.Equal(model.DataOffset, tensor.AbsoluteOffset)

[<Fact>]
let ``empty tensors may share an offset with a nonempty tensor`` () =
    let descriptors = [ tensor false "a" [ 0UL ] 0u 0UL
                        tensor false "b" [ 1UL ] 0u 0UL
                        tensor false "c" [ 0UL ] 0u 0UL ]
    let model = parse (file false 3u 32 [] descriptors (Array.zeroCreate 4))
    Assert.Equal(3, model.Tensors.Length)

type private GuardedStream(bytes: byte array, dataOffset: int64) =
    inherit MemoryStream(bytes, false)
    override this.Read(buffer: byte array, offset: int, count: int) =
        if this.Position + int64 count > dataOffset then failwith "Parser attempted to read weights."
        base.Read(buffer, offset, min count 1)
    override this.Read(buffer: Span<byte>) =
        if this.Position + int64 buffer.Length > dataOffset then failwith "Parser attempted to read weights."
        base.Read(buffer.Slice(0, min buffer.Length 1))

[<Fact>]
let ``handles partial reads and leaves caller stream open at the data section`` () =
    let bytes = file false 3u 32 [] [ tensor false "weight" [ 2UL ] 0u 0UL ] (Array.zeroCreate 8)
    let expectedDataOffset = int64 bytes.Length - 8L
    use stream = new GuardedStream(bytes, expectedDataOffset)
    let model = Gguf.read stream
    Assert.Equal(expectedDataOffset, model.DataOffset)
    Assert.Equal(expectedDataOffset, stream.Position)
    Assert.True(stream.CanRead)

[<Fact>]
let ``reads an embedded GGUF relative to the initial stream position`` () =
    let bytes = file false 3u 32 [] [ tensor false "weight" [ 1UL ] 0u 0UL ] (u32 false 0x3f800000u)
    use stream = new MemoryStream(concat [ Array.zeroCreate 7; bytes ], false)
    stream.Position <- 7L
    let model = Gguf.read stream
    Assert.Equal(int64 bytes.Length + 3L, model.DataOffset)
    Assert.Equal(model.DataOffset, model.Tensors[0].AbsoluteOffset)
    Assert.Equal(0x3f800000u, (new BinaryReader(stream, Encoding.UTF8, true)).ReadUInt32())

[<Fact>]
let ``readFile closes its file handle`` () =
    let path = Path.GetTempFileName()
    try
        File.WriteAllBytes(path, file false 2u 32 [] [] [||])
        Assert.Equal(2u, (Gguf.readFile path).Version)
        use exclusive = File.Open(path, FileMode.Open, FileAccess.ReadWrite, FileShare.None)
        Assert.True(exclusive.CanWrite)
    finally File.Delete(path)

[<Fact>]
let ``rejects every truncated prefix of a valid file`` () =
    let metadata = [ kv false "general.name" 8u (str false "Marmotte")
                     kv false "tokenizer.scores" 9u (arr false 6u [ u32 false 0x3f800000u ]) ]
    let bytes = file false 3u 32 metadata [ tensor false "weight" [ 2UL ] 0u 0UL ] (Array.zeroCreate 8)
    for length = 0 to bytes.Length - 1 do
        invalid bytes[..length - 1] |> ignore

[<Theory>]
[<InlineData(0u)>]
[<InlineData(1u)>]
[<InlineData(4u)>]
[<InlineData(0x02000000u)>]
let ``rejects unsupported versions`` version =
    invalid (file false version 32 [] [] [||]) |> ignore

[<Fact>]
let ``rejects invalid magic`` () =
    let bytes = file false 3u 32 [] [] [||]
    bytes[0] <- 0uy
    invalid bytes |> ignore

[<Theory>]
[<InlineData(0u)>]
[<InlineData(1u)>]
[<InlineData(7u)>]
[<InlineData(12u)>]
let ``rejects invalid alignment values`` alignment =
    invalid (file false 3u 32 [ kv false "general.alignment" 4u (u32 false alignment) ] [] [||]) |> ignore

[<Fact>]
let ``rejects alignment with the wrong metadata type`` () =
    invalid (file false 3u 32 [ kv false "general.alignment" 10u (u64 false 32UL) ] [] [||]) |> ignore

[<Theory>]
[<InlineData(13u)>]
[<InlineData(0xffffffffu)>]
let ``rejects unknown metadata and empty array element types`` tag =
    invalid (file false 3u 32 [ kv false "test.value" tag [||] ] [] [||]) |> ignore
    invalid (file false 3u 32 [ kv false "test.value" 9u (arr false tag []) ] [] [||]) |> ignore

[<Fact>]
let ``rejects invalid boolean and UTF8`` () =
    invalid (file false 3u 32 [ kv false "test.value" 7u [| 2uy |] ] [] [||]) |> ignore
    let badString = concat [ u64 false 2UL; [| 0xc3uy; 0x28uy |] ]
    invalid (file false 3u 32 [ kv false "test.value" 8u badString ] [] [||]) |> ignore

[<Theory>]
[<InlineData("")>]
[<InlineData("general.Name")>]
[<InlineData("general..name")>]
[<InlineData("general.λ")>]
[<InlineData("general.name.")>]
let ``rejects malformed metadata keys`` key =
    invalid (file false 3u 32 [ kv false key 0u [| 0uy |] ] [] [||]) |> ignore

[<Fact>]
let ``accepts numeric metadata key segments`` () =
    let model = parse (file false 3u 32 [ kv false "general.base_model.0.name" 8u (str false "model") ] [] [||])
    Assert.Equal(GgufValue.String "model", model.Metadata["general.base_model.0.name"])

[<Fact>]
let ``rejects duplicate keys and tensor names`` () =
    let entry = kv false "test.value" 0u [| 0uy |]
    invalid (file false 3u 32 [ entry; entry ] [] [||]) |> ignore
    invalid (file false 3u 32 [] [ tensor false "weight" [ 1UL ] 0u 0UL; tensor false "weight" [ 1UL ] 0u 32UL ] (Array.zeroCreate 36)) |> ignore

[<Fact>]
let ``rejects oversized counts and lengths before allocation`` () =
    invalid (header false 3u UInt64.MaxValue 0UL) |> ignore
    invalid (header false 3u 0UL UInt64.MaxValue) |> ignore
    invalid (file false 3u 32 [ kv false "test.value" 8u (u64 false UInt64.MaxValue) ] [] [||]) |> ignore
    invalid (file false 3u 32 [ kv false "test.value" 9u (concat [ u32 false 0u; u64 false UInt64.MaxValue ]) ] [] [||]) |> ignore

[<Fact>]
let ``enforces key and tensor name byte limits`` () =
    invalid (file false 3u 32 [ kv false (String('a', 65536)) 0u [| 0uy |] ] [] [||]) |> ignore
    invalid (file false 3u 32 [] [ tensor false (String('λ', 33)) [ 1UL ] 0u 0UL ] (Array.zeroCreate 4)) |> ignore
    let model = parse (file false 3u 32 [] [ tensor false (String('λ', 32)) [ 1UL ] 0u 0UL ] (Array.zeroCreate 4))
    Assert.Equal(32, model.Tensors[0].Name.Length)

[<Fact>]
let ``rejects invalid shapes and size overflows`` () =
    for dimensions, tag in [ [ 1UL; 1UL; 1UL; 1UL; 1UL ], 0u
                             [ UInt64.MaxValue; 2UL ], 0u
                             [ UInt64.MaxValue ], 0u; [ 16UL; 2UL ], 2u ] do
        invalid (file false 3u 32 [] [ tensor false "weight" dimensions tag 0UL ] (Array.zeroCreate 64)) |> ignore

[<Fact>]
let ``rejects invalid tensor offsets and overlapping payloads`` () =
    for offset in [ 1UL; 64UL; 0xffffffffffffffe0UL ] do
        invalid (file false 3u 32 [] [ tensor false "weight" [ 1UL ] 0u offset ] (Array.zeroCreate 64)) |> ignore
    for offset in [ 0UL; 32UL ] do
        let descriptors = [ tensor false "a" [ 16UL ] 0u 0UL; tensor false "b" [ 1UL ] 0u offset ]
        invalid (file false 3u 32 [] descriptors (Array.zeroCreate 64)) |> ignore

[<Fact>]
let ``enforces configurable read limits`` () =
    let defaults = GgufReadOptions.Default
    let metadata = [ kv false "a" 9u (arr false 8u [ str false "long"; str false "text" ]) ]
    let bytes = file false 3u 32 metadata [ tensor false "weight" [ 1UL ] 0u 0UL ] (Array.zeroCreate 4)
    let limited = [ { defaults with MaxStringBytes = 3 }; { defaults with MaxArrayElements = 1 }
                    { defaults with MaxMetadataEntries = 0 }; { defaults with MaxTensors = 0 }
                    { defaults with MaxArrayDepth = 0 }; { defaults with MaxMetadataValues = 2 }
                    { defaults with MaxHeaderBytes = 24L } ]
    for options in limited do
        Assert.Throws<InvalidDataException>(fun () -> parseWith options bytes |> ignore) |> ignore
    let model = parseWith { defaults with MaxMetadataValues = 3 } bytes
    Assert.Single(model.Tensors) |> ignore

[<Fact>]
let ``bounds recursive arrays and the total value budget`` () =
    let rec nested depth =
        if depth = 1 then arr false 0u [ [| 1uy |] ]
        else arr false 9u [ nested (depth - 1) ]
    let fixture depth = file false 3u 32 [ kv false "a" 9u (nested depth) ] [] [||]
    parse (fixture 16) |> ignore
    invalid (fixture 17) |> ignore
    let arrays = [ arr false 0u [ [| 1uy |]; [| 2uy |] ]; arr false 0u [ [| 3uy |]; [| 4uy |] ] ]
    let bytes = file false 3u 32 [ kv false "a" 9u (arr false 9u arrays) ] [] [||]
    Assert.Throws<InvalidDataException>(fun () -> parseWith { GgufReadOptions.Default with MaxMetadataValues = 6 } bytes |> ignore) |> ignore

[<Fact>]
let ``validates options and stream capabilities`` () =
    let defaults = GgufReadOptions.Default
    let bytes = file false 3u 32 [] [] [||]
    for options in [ { defaults with MaxStringBytes = -1 }; { defaults with MaxArrayElements = -1 }
                     { defaults with MaxMetadataEntries = -1 }; { defaults with MaxTensors = -1 }
                     { defaults with MaxArrayDepth = -1 }; { defaults with MaxArrayDepth = 65 }
                     { defaults with MaxMetadataValues = -1 }; { defaults with MaxHeaderBytes = 23L } ] do
        Assert.Throws<ArgumentException>(fun () -> parseWith options bytes |> ignore) |> ignore
    Assert.Throws<ArgumentNullException>(fun () -> Gguf.read null |> ignore) |> ignore
    use nonSeekable =
        { new MemoryStream(bytes) with
            override _.CanSeek = false }
    Assert.Throws<ArgumentException>(fun () -> Gguf.read nonSeekable |> ignore) |> ignore
    let closed = new MemoryStream(bytes)
    closed.Dispose()
    Assert.Throws<ArgumentException>(fun () -> Gguf.read closed |> ignore) |> ignore
