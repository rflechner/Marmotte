module Marmotte.Safetensors.Tests

open System
open System.IO
open System.Text
open Marmotte.Safetensors
open Xunit

// Fixtures encode the format directly: an 8-byte little-endian header length, the UTF-8
// JSON header, then the tensor data buffer. They do not reuse the parser's dtype table.
let private littleEndian64 (value: uint64) =
    Array.init 8 (fun index -> byte (value >>> (index * 8)))

let private fileFromHeaderBytes (header: byte array) (data: byte array) =
    Array.concat [ littleEndian64 (uint64 header.Length); header; data ]

let private file (header: string) data =
    fileFromHeaderBytes (Encoding.UTF8.GetBytes header) data

let private tensor name dtype (shape: uint64 list) (startOffset: uint64) (endOffset: uint64) =
    let dimensions = shape |> List.map string |> String.concat ","
    sprintf "\"%s\":{\"dtype\":\"%s\",\"shape\":[%s],\"data_offsets\":[%d,%d]}" name dtype dimensions startOffset endOffset

let private header (entries: string list) =
    "{" + String.concat "," entries + "}"

// The worked example of docs/safetensors.md: a 112-byte header followed by 36 data bytes.
let private workedExampleHeader =
    """{"a":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]},"b":{"dtype":"I32","shape":[3],"data_offsets":[24,36]}}"""

let private workedExample () =
    file workedExampleHeader (Array.init 36 byte)

let private parse bytes =
    use stream = new MemoryStream(bytes, false)
    Safetensors.read stream

let private parseWith options bytes =
    use stream = new MemoryStream(bytes, false)
    Safetensors.readWithOptions options stream

let private invalid bytes =
    Assert.Throws<InvalidDataException>(fun () -> parse bytes |> ignore)

/// Returns at most one byte per read and fails if the parser reads tensor data.
type private GuardedStream(bytes: byte array, dataOffset: int64) =
    inherit MemoryStream(bytes, false)

    override this.Read(buffer: byte array, offset: int, count: int) =
        if this.Position + int64 count > dataOffset then failwith "Parser attempted to read weights."
        base.Read(buffer, offset, min count 1)

    override this.Read(buffer: Span<byte>) =
        if this.Position + int64 buffer.Length > dataOffset then failwith "Parser attempted to read weights."
        base.Read(buffer.Slice(0, min buffer.Length 1))

[<Fact>]
let ``reads the worked example with partial reads and leaves the stream at the data buffer`` () =
    let bytes = workedExample ()
    Assert.Equal(156, bytes.Length)

    use stream = new GuardedStream(bytes, 120L)
    let model = Safetensors.read stream

    Assert.Equal(112UL, model.HeaderLength)
    Assert.Equal(120L, model.DataOffset)
    Assert.Equal(36UL, model.DataLength)
    Assert.Equal(120L, stream.Position)
    Assert.True(stream.CanRead)
    Assert.Empty(model.Metadata)

    let first = model.Tensors[0]
    Assert.Equal("a", first.Name)
    Assert.Equal(SafetensorsDtype.F32, first.Dtype)
    Assert.Equal<uint64>([| 2UL; 3UL |], first.Shape)
    Assert.Equal(6UL, first.ElementCount)
    Assert.Equal(0UL, first.Offset)
    Assert.Equal(24UL, first.ByteSize)
    Assert.Equal(120L, first.AbsoluteOffset)

    let second = model.Tensors[1]
    Assert.Equal("b", second.Name)
    Assert.Equal(SafetensorsDtype.I32, second.Dtype)
    Assert.Equal(24UL, second.Offset)
    Assert.Equal(12UL, second.ByteSize)
    Assert.Equal(144L, second.AbsoluteOffset)

[<Fact>]
let ``orders tensors by data offset and reads metadata scalars and empty tensors`` () =
    let json =
        header [ tensor "z.scalar" "F64" [] 8UL 16UL
                 "\"__metadata__\":{\"format\":\"pt\",\"note\":\"Unicode λ 🦫\"}"
                 tensor "empty.late" "BF16" [ 0UL; 4UL ] 16UL 16UL
                 tensor "a.first" "U8" [ 8UL ] 0UL 8UL
                 tensor "empty.early" "F32" [ 4UL; 0UL ] 0UL 0UL ]
        + "  \n\t"

    let model = parse (file json (Array.zeroCreate 16))

    let names = model.Tensors |> Array.map (fun tensor -> tensor.Name)
    Assert.Equal<string>([| "empty.early"; "a.first"; "z.scalar"; "empty.late" |], names)
    Assert.Equal<Map<string, string>>(Map [ "format", "pt"; "note", "Unicode λ 🦫" ], model.Metadata)
    Assert.Equal(uint64 (Encoding.UTF8.GetByteCount json), model.HeaderLength)

    let scalar = model.Tensors[2]
    Assert.Empty(scalar.Shape)
    Assert.Equal(1UL, scalar.ElementCount)
    Assert.Equal(8UL, scalar.ByteSize)

    for empty in [ model.Tensors[0]; model.Tensors[3] ] do
        Assert.Equal(0UL, empty.ElementCount)
        Assert.Equal(0UL, empty.ByteSize)

[<Theory>]
[<InlineData("{}")>]
[<InlineData("""{"__metadata__":{}}""")>]
let ``accepts headers without tensors and an empty data buffer`` json =
    let model = parse (file json [||])
    Assert.Empty(model.Tensors)
    Assert.Equal(0UL, model.DataLength)

// Widths from Dtype::bitsize in https://github.com/huggingface/safetensors (checked 2026-10-24).
[<Theory>]
[<InlineData("BOOL", 8)>]
[<InlineData("F4", 4)>]
[<InlineData("F6_E2M3", 6)>]
[<InlineData("F6_E3M2", 6)>]
[<InlineData("U8", 8)>]
[<InlineData("I8", 8)>]
[<InlineData("F8_E5M2", 8)>]
[<InlineData("F8_E4M3", 8)>]
[<InlineData("F8_E8M0", 8)>]
[<InlineData("F8_E4M3FNUZ", 8)>]
[<InlineData("F8_E5M2FNUZ", 8)>]
[<InlineData("I16", 16)>]
[<InlineData("U16", 16)>]
[<InlineData("F16", 16)>]
[<InlineData("BF16", 16)>]
[<InlineData("I32", 32)>]
[<InlineData("U32", 32)>]
[<InlineData("F32", 32)>]
[<InlineData("C64", 64)>]
[<InlineData("F64", 64)>]
[<InlineData("I64", 64)>]
[<InlineData("U64", 64)>]
let ``accepts every reference dtype with its bit width`` wireName (bits: int) =
    // Eight elements fill whole bytes for every width, including 4- and 6-bit dtypes.
    let byteCount = uint64 bits
    let json = header [ tensor "t" wireName [ 8UL ] 0UL byteCount ]
    let parsed = (parse (file json (Array.zeroCreate bits))).Tensors[0]

    Assert.Equal(wireName, parsed.Dtype.WireName)
    Assert.Equal(uint64 bits, parsed.Dtype.BitsPerElement)
    Assert.Equal(byteCount, parsed.ByteSize)

[<Fact>]
let ``packs sub-byte elements into whole bytes`` () =
    let model = parse (file (header [ tensor "packed" "F6_E2M3" [ 4UL ] 0UL 3UL ]) (Array.zeroCreate 3))
    Assert.Equal(3UL, model.Tensors[0].ByteSize)

[<Theory>]
[<InlineData("F4", 3)>]
[<InlineData("F6_E2M3", 2)>]
[<InlineData("F6_E3M2", 5)>]
let ``rejects sub-byte tensors that do not fill whole bytes`` dtype (elements: int) =
    let json = header [ tensor "packed" dtype [ uint64 elements ] 0UL 1UL ]
    let error = invalid (file json [| 0uy |])
    Assert.Contains("whole number of bytes", error.Message)

[<Fact>]
let ``reads an embedded file relative to the initial stream position`` () =
    use stream = new MemoryStream(Array.append (Array.zeroCreate 5) (workedExample ()), false)
    stream.Position <- 5L

    let model = Safetensors.read stream

    Assert.Equal(125L, model.DataOffset)
    Assert.Equal(149L, model.Tensors[1].AbsoluteOffset)
    Assert.Equal(125L, stream.Position)

[<Fact>]
let ``readFile closes its file handle`` () =
    let path = Path.GetTempFileName()
    try
        File.WriteAllBytes(path, workedExample ())
        Assert.Equal(2, (Safetensors.readFile path).Tensors.Length)
        use exclusive = File.Open(path, FileMode.Open, FileAccess.ReadWrite, FileShare.None)
        Assert.True(exclusive.CanWrite)
    finally
        File.Delete(path)

[<Fact>]
let ``rejects every truncated prefix of a valid file`` () =
    let bytes = workedExample ()
    for length = 0 to bytes.Length - 1 do
        invalid bytes[..length - 1] |> ignore

[<Theory>]
[<InlineData(28, 40, 40, "4 bytes before tensor 'b' belong to no tensor")>]
[<InlineData(20, 32, 32, "Tensor 'b' overlaps")>]
[<InlineData(24, 36, 37, "1 bytes at the end of the tensor data belong to no tensor")>]
[<InlineData(24, 36, 35, "beyond the 35 bytes remaining")>]
let ``requires tensors to cover the data buffer exactly`` (startOffset: int) (endOffset: int) (dataLength: int) (expected: string) =
    let json =
        header [ tensor "a" "F32" [ 2UL; 3UL ] 0UL 24UL
                 tensor "b" "I32" [ 3UL ] (uint64 startOffset) (uint64 endOffset) ]

    let error = invalid (file json (Array.zeroCreate dataLength))
    Assert.Contains(expected, error.Message)

[<Fact>]
let ``empty tensors cannot start beyond the covered data`` () =
    let json =
        header [ tensor "a" "F32" [ 2UL; 3UL ] 0UL 24UL
                 tensor "empty" "F32" [ 0UL ] 30UL 30UL ]

    let error = invalid (file json (Array.zeroCreate 24))
    Assert.Contains("before tensor 'empty' belong to no tensor", error.Message)

[<Theory>]
[<InlineData("\"t\":{\"dtype\":\"F128\",\"shape\":[1],\"data_offsets\":[0,4]}", "unsupported dtype 'F128'")>]
[<InlineData("\"t\":{\"dtype\":\"f32\",\"shape\":[1],\"data_offsets\":[0,4]}", "unsupported dtype 'f32'")>]
[<InlineData("\"t\":{\"dtype\":32,\"shape\":[1],\"data_offsets\":[0,4]}", "dtype must be a string")>]
[<InlineData("\"t\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4],\"strides\":[1]}", "unknown field 'strides'")>]
[<InlineData("\"t\":{\"dtype\":\"F32\",\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}", "duplicate field 'dtype'")>]
[<InlineData("\"t\":{\"dtype\":\"F32\",\"data_offsets\":[0,4]}", "no shape field")>]
[<InlineData("\"t\":{\"dtype\":\"F32\",\"shape\":[1]}", "no data_offsets field")>]
[<InlineData("\"t\":[1]", "must be described by a JSON object")>]
[<InlineData("\"t\":{\"dtype\":\"F32\",\"shape\":1,\"data_offsets\":[0,4]}", "shape must be an array")>]
[<InlineData("\"t\":{\"dtype\":\"F32\",\"shape\":[-1],\"data_offsets\":[0,4]}", "dimension 0 must be a nonnegative integer")>]
[<InlineData("\"t\":{\"dtype\":\"F32\",\"shape\":[1.5],\"data_offsets\":[0,4]}", "dimension 0 must be a nonnegative integer")>]
[<InlineData("\"t\":{\"dtype\":\"F32\",\"shape\":[1e0],\"data_offsets\":[0,4]}", "dimension 0 must be a nonnegative integer")>]
[<InlineData("\"t\":{\"dtype\":\"F32\",\"shape\":[\"1\"],\"data_offsets\":[0,4]}", "dimension 0 must be a nonnegative integer")>]
[<InlineData("\"t\":{\"dtype\":\"F32\",\"shape\":[1,18446744073709551616],\"data_offsets\":[0,4]}", "dimension 1 must be a nonnegative integer")>]
[<InlineData("\"t\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0]}", "array of two integers")>]
[<InlineData("\"t\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4,8]}", "array of two integers")>]
[<InlineData("\"t\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,-4]}", "data_offsets end must be a nonnegative integer")>]
[<InlineData("\"t\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[4,0]}", "end (0) precedes its start (4)")>]
[<InlineData("\"t\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,8]}", "cover 8 bytes; its shape and dtype require 4")>]
[<InlineData("\"t\":{\"dtype\":\"F32\",\"shape\":[4294967296,4294967296,2],\"data_offsets\":[0,4]}", "size overflows uint64")>]
let ``rejects invalid tensor descriptors`` entry (expected: string) =
    let error = invalid (file ("{" + entry + "}") (Array.zeroCreate 4))
    Assert.Contains(expected, error.Message)

[<Theory>]
[<InlineData("""{"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}""", "Duplicate key 'a'")>]
[<InlineData("""{"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}""", "Duplicate key 'a'")>]
[<InlineData("""{"__metadata__":{},"__metadata__":{}}""", "Duplicate key '__metadata__'")>]
[<InlineData("""{"__metadata__":{"k":"1","k":"2"}}""", "Duplicate __metadata__ key 'k'")>]
let ``rejects duplicate keys after unescaping`` json (expected: string) =
    let error = invalid (file json [| 0uy |])
    Assert.Contains(expected, error.Message)

[<Theory>]
[<InlineData("""{"__metadata__":{"n":1}}""")>]
[<InlineData("""{"__metadata__":{"n":null}}""")>]
[<InlineData("""{"__metadata__":["format"]}""")>]
[<InlineData("""{"__metadata__":null}""")>]
let ``requires metadata to be an object of strings`` json =
    let error = invalid (file json [||])
    Assert.Contains("__metadata__", error.Message)

[<Theory>]
[<InlineData("", "must start with '{'")>]
[<InlineData(" {}", "must start with '{'")>]
[<InlineData("[]", "must start with '{'")>]
[<InlineData("{}x", "after a single JSON value")>]
[<InlineData("{}\u0000", "after a single JSON value")>]
[<InlineData("{\"a\":{},}", "Invalid Safetensors header")>]
[<InlineData("{/* comment */}", "Invalid Safetensors header")>]
[<InlineData("""{"a":{"shape":[[1]]}}""", "maximum configured depth")>]
[<InlineData("""{"\ud800":{}}""", "surrogate")>]
let ``rejects malformed JSON headers`` json (expected: string) =
    let error = invalid (file json [||])
    Assert.Contains(expected, error.Message)

[<Fact>]
let ``rejects headers that are not valid UTF-8`` () =
    let headerBytes = Array.concat [ "{\""B; [| 0xFFuy |]; "\":{}}"B ]
    let error = invalid (fileFromHeaderBytes headerBytes [||])
    Assert.Contains("not valid UTF-8", error.Message)

[<Fact>]
let ``rejects header lengths beyond the file or the configured limit`` () =
    let bytes = workedExample ()
    let withHeaderLength (length: uint64) = Array.append (littleEndian64 length) bytes[8..]

    Assert.Contains("remaining file size", (invalid (withHeaderLength 1000UL)).Message)
    Assert.Contains("MaxHeaderBytes", (invalid (withHeaderLength UInt64.MaxValue)).Message)

    let limited = { SafetensorsReadOptions.Default with MaxHeaderBytes = 111 }
    let error = Assert.Throws<InvalidDataException>(fun () -> parseWith limited bytes |> ignore)
    Assert.Contains("MaxHeaderBytes", error.Message)

    let exact = { SafetensorsReadOptions.Default with MaxHeaderBytes = 112 }
    Assert.Equal(2, (parseWith exact bytes).Tensors.Length)

[<Fact>]
let ``enforces configurable tensor and rank limits`` () =
    let bytes = workedExample ()
    let withLimits maxTensors maxRank =
        { SafetensorsReadOptions.Default with MaxTensors = maxTensors; MaxRank = maxRank }

    let tooMany = Assert.Throws<InvalidDataException>(fun () -> parseWith (withLimits 1 2) bytes |> ignore)
    Assert.Contains("MaxTensors", tooMany.Message)

    let tooDeep = Assert.Throws<InvalidDataException>(fun () -> parseWith (withLimits 2 1) bytes |> ignore)
    Assert.Contains("MaxRank", tooDeep.Message)

    Assert.Equal(2, (parseWith (withLimits 2 2) bytes).Tensors.Length)
    Assert.Throws<ArgumentException>(fun () -> parseWith (withLimits 2 -1) bytes |> ignore) |> ignore

module TensorSource =
    open Marmotte.Core

    let private withFile (bytes: byte array) action =
        let path = Path.GetTempFileName()
        try
            File.WriteAllBytes(path, bytes)
            action path
        finally
            File.Delete(path)

    [<Fact>]
    let ``keeps row-major shapes and reads payloads from the data buffer`` () =
        withFile (workedExample ()) (fun path ->
            use source = SafetensorsTensorSource.openFile path
            Assert.Equal<string>([| "a"; "b" |], source.Tensors |> Seq.map (fun tensor -> tensor.Name))

            let descriptor = source.Tensors[0]
            Assert.Equal<uint64>([| 2UL; 3UL |], descriptor.Shape)
            Assert.Equal(Some TensorEncoding.F32, descriptor.Encoding)
            Assert.Equal("F32", descriptor.SourceEncoding)
            Assert.Equal(Some 24UL, descriptor.ByteSize)

            // The data buffer of the worked example holds the bytes 0 to 35.
            let matrix = WeightMatrix.Read(source, "a")
            Assert.Equal(2, matrix.Rows)
            Assert.Equal(3, matrix.Columns)
            Assert.Equal<byte>(Array.init 12 (fun index -> byte (12 + index)), matrix.Row(1).ToArray())

            Assert.Equal("I32", source.Tensors[1].SourceEncoding)
            Assert.Equal(None, source.Tensors[1].Encoding)
            let error = Assert.Throws<InvalidDataException>(fun () -> source.ReadPayload("b") |> ignore)
            Assert.Contains("I32", error.Message))

    [<Fact>]
    let ``maps only floating-point dtypes to runtime encodings`` () =
        Assert.Equal(Some TensorEncoding.F32, SafetensorsTensorSource.tryEncoding SafetensorsDtype.F32)
        Assert.Equal(Some TensorEncoding.F16, SafetensorsTensorSource.tryEncoding SafetensorsDtype.F16)
        Assert.Equal(Some TensorEncoding.BF16, SafetensorsTensorSource.tryEncoding SafetensorsDtype.BF16)

        for dtype in [ SafetensorsDtype.F8_E4M3; SafetensorsDtype.F64; SafetensorsDtype.I8; SafetensorsDtype.U8 ] do
            Assert.Equal(None, SafetensorsTensorSource.tryEncoding dtype)

    [<Fact>]
    let ``closes the file when its index is invalid`` () =
        let bytes = workedExample ()

        withFile bytes[..bytes.Length - 2] (fun path ->
            Assert.Throws<InvalidDataException>(fun () -> SafetensorsTensorSource.openFile path |> ignore) |> ignore
            use exclusive = File.Open(path, FileMode.Open, FileAccess.ReadWrite, FileShare.None)
            Assert.True(exclusive.CanWrite))
