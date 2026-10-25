module Marmotte.Core.Tests

open System
open System.Collections.Generic
open System.IO
open Marmotte.Core
open Xunit

type private StoredTensor =
    { Descriptor: TensorDescriptor
      Payload: byte array }

/// Serves payloads from memory and counts reads, so matrix contracts are tested without a file format.
type private InMemoryTensorSource(tensors: StoredTensor list) =
    let mutable reads = 0
    let find name = tensors |> List.tryFind (fun tensor -> tensor.Descriptor.Name = name)

    member _.Reads = reads

    interface ITensorSource with
        member _.Tensors =
            tensors |> List.map (fun tensor -> tensor.Descriptor) |> Array.ofList :> IReadOnlyList<TensorDescriptor>

        member _.TryFindTensor(name) =
            find name |> Option.map (fun tensor -> tensor.Descriptor)

        member _.ReadPayload(name) =
            reads <- reads + 1
            match find name with
            | Some tensor -> ReadOnlyMemory<byte>(tensor.Payload)
            | None -> raise (KeyNotFoundException(name))

    interface IDisposable with
        member _.Dispose() = ()

// Unsupported tensors use I32, a type that neither format maps to a runtime encoding.
let private descriptor name (encoding: TensorEncoding option) (shape: uint64 list) (byteSize: int) =
    { Name = name
      Encoding = encoding
      SourceEncoding = encoding |> Option.map string |> Option.defaultValue "I32"
      Shape = Array.ofList shape
      ElementCount = List.fold (*) 1UL shape
      ByteSize = Some(uint64 byteSize) }

let private stored name encoding shape (payload: byte array) =
    { Descriptor = descriptor name encoding shape payload.Length
      Payload = payload }

let private sourceOf tensors = new InMemoryTensorSource(tensors)

// Block sizes from the quantized block layouts cited in docs/gguf.md.
let private encodingNamed name =
    match name with
    | "F32" -> TensorEncoding.F32
    | "F16" -> TensorEncoding.F16
    | "BF16" -> TensorEncoding.BF16
    | "Q4_0" -> TensorEncoding.Q4_0
    | "Q4_1" -> TensorEncoding.Q4_1
    | "Q5_0" -> TensorEncoding.Q5_0
    | "Q5_1" -> TensorEncoding.Q5_1
    | "Q8_0" -> TensorEncoding.Q8_0
    | "IQ4_NL" -> TensorEncoding.IQ4_NL
    | "Q2_K" -> TensorEncoding.Q2_K
    | "Q3_K" -> TensorEncoding.Q3_K
    | "Q4_K" -> TensorEncoding.Q4_K
    | "Q5_K" -> TensorEncoding.Q5_K
    | "Q6_K" -> TensorEncoding.Q6_K
    | _ -> failwith $"Unknown test encoding {name}"

[<Fact>]
let ``reads matrix rows in row-major order`` () =
    let payload = Array.init 24 byte
    use source = sourceOf [ stored "matrix" (Some TensorEncoding.F32) [ 2UL; 3UL ] payload ]

    let matrix = WeightMatrix.Read(source, "matrix")

    Assert.Equal("matrix", matrix.Name)
    Assert.Equal(TensorEncoding.F32, matrix.Encoding)
    Assert.Equal(2, matrix.Rows)
    Assert.Equal(3, matrix.Columns)
    Assert.Equal(12, matrix.RowBytes)
    Assert.Equal<byte>(payload[..11], matrix.Row(0).ToArray())
    Assert.Equal<byte>(payload[12..], matrix.Row(1).ToArray())

[<Theory>]
[<InlineData("F16", 3, 6)>]
[<InlineData("BF16", 5, 10)>]
[<InlineData("Q4_0", 64, 36)>]
[<InlineData("Q4_1", 64, 40)>]
[<InlineData("Q5_0", 32, 22)>]
[<InlineData("Q5_1", 64, 48)>]
[<InlineData("Q8_0", 32, 34)>]
[<InlineData("IQ4_NL", 96, 54)>]
[<InlineData("Q2_K", 256, 84)>]
[<InlineData("Q3_K", 512, 220)>]
[<InlineData("Q4_K", 256, 144)>]
[<InlineData("Q5_K", 256, 176)>]
[<InlineData("Q6_K", 512, 420)>]
let ``rows hold whole quantization blocks`` encodingName columns rowBytes =
    let payload = Array.init (3 * rowBytes) byte
    let encoding = encodingNamed encodingName
    use source = sourceOf [ stored "matrix" (Some encoding) [ 3UL; uint64 columns ] payload ]

    let matrix = WeightMatrix.Read(source, "matrix")

    Assert.Equal(rowBytes, matrix.RowBytes)
    Assert.Equal<byte>(payload[2 * rowBytes ..], matrix.Row(2).ToArray())

[<Fact>]
let ``checks the shape before reading the payload`` () =
    let tensors =
        [ stored "unsupported" None [ 2UL; 2UL ] (Array.zeroCreate 16)
          stored "vector" (Some TensorEncoding.F32) [ 4UL ] (Array.zeroCreate 16)
          stored "cube" (Some TensorEncoding.F32) [ 1UL; 2UL; 2UL ] (Array.zeroCreate 16)
          stored "no-rows" (Some TensorEncoding.F32) [ 0UL; 4UL ] [||]
          stored "partial-block" (Some TensorEncoding.Q8_0) [ 2UL; 48UL ] (Array.zeroCreate 68) ]
    use source = sourceOf tensors

    let expectedMessages =
        [ "unsupported", "I32"
          "vector", "rank 1"
          "cube", "rank 3"
          "no-rows", "0 rows"
          "partial-block", "Q8_0 blocks" ]
    for name, expected in expectedMessages do
        let error = Assert.Throws<InvalidDataException>(fun () -> WeightMatrix.Read(source, name) |> ignore)
        Assert.Contains(expected, error.Message)

    Assert.Equal(0, source.Reads)

[<Fact>]
let ``rejects payloads that do not match the shape`` () =
    let tensor =
        { Descriptor = descriptor "matrix" (Some TensorEncoding.F32) [ 2UL; 3UL ] 24
          Payload = Array.zeroCreate 20 }
    use source = sourceOf [ tensor ]

    let error = Assert.Throws<InvalidDataException>(fun () -> WeightMatrix.Read(source, "matrix") |> ignore)
    Assert.Contains("20-byte payload", error.Message)

[<Fact>]
let ``rejects unknown tensors and rows out of range`` () =
    use source = sourceOf [ stored "matrix" (Some TensorEncoding.F32) [ 2UL; 1UL ] (Array.zeroCreate 8) ]

    Assert.Throws<KeyNotFoundException>(fun () -> WeightMatrix.Read(source, "missing") |> ignore) |> ignore

    let matrix = WeightMatrix.Read(source, "matrix")
    Assert.Throws<ArgumentOutOfRangeException>(fun () -> matrix.Row(-1).Length |> ignore) |> ignore
    Assert.Throws<ArgumentOutOfRangeException>(fun () -> matrix.Row(2).Length |> ignore) |> ignore

[<Fact>]
let ``decodes one matrix row into exactly Columns values`` () =
    let values = [| 1.0f; 2.0f; 3.0f; 4.0f; 5.0f; 6.0f |]
    let payload = values |> Array.collect BitConverter.GetBytes
    use source = sourceOf [ stored "matrix" (Some TensorEncoding.F32) [ 2UL; 3UL ] payload ]
    let matrix = WeightMatrix.Read(source, "matrix")
    let decoded = Array.zeroCreate<single> 3

    matrix.DecodeRow(1, Span<single>(decoded))

    Assert.Equal<single>(values[3..], decoded)
    Assert.Throws<ArgumentException>(fun () -> matrix.DecodeRow(0, Span<single>(Array.zeroCreate 2))) |> ignore
    Assert.Throws<ArgumentOutOfRangeException>(fun () -> matrix.DecodeRow(2, Span<single>(decoded))) |> ignore

let private withFile (bytes: byte array) action =
    let path = Path.GetTempFileName()
    try
        File.WriteAllBytes(path, bytes)
        action path
    finally
        File.Delete(path)

let private location name encoding shape byteSize offset =
    { Descriptor = descriptor name encoding shape byteSize
      AbsoluteOffset = offset }

[<Fact>]
let ``file source copies payloads from absolute offsets`` () =
    let bytes = Array.init 64 byte
    let locations =
        [| location "matrix" (Some TensorEncoding.F32) [ 2UL; 2UL ] 16 8L
           location "indices" None [ 1UL ] 4 32L |]

    withFile bytes (fun path ->
        use source = new FileTensorSource(File.OpenRead(path), locations) :> ITensorSource

        Assert.Equal<string>([| "matrix"; "indices" |], source.Tensors |> Seq.map (fun tensor -> tensor.Name))
        Assert.Equal(Some [| 1UL |], source.TryFindTensor("indices") |> Option.map (fun tensor -> tensor.Shape))
        Assert.Equal(None, source.TryFindTensor("Matrix"))
        Assert.Equal<byte>(bytes[8..23], source.ReadPayload("matrix").ToArray())

        let error = Assert.Throws<InvalidDataException>(fun () -> source.ReadPayload("indices") |> ignore)
        Assert.Contains("I32", error.Message)
        Assert.Throws<KeyNotFoundException>(fun () -> source.ReadPayload("missing") |> ignore) |> ignore)

[<Fact>]
let ``file source rejects duplicate names and ranges outside the file`` () =
    let matrix = location "matrix" (Some TensorEncoding.F32) [ 1UL; 2UL ] 8 0L
    let invalidLocations =
        [ [| matrix; matrix |]
          [| { matrix with AbsoluteOffset = 9L } |]
          [| { matrix with AbsoluteOffset = -1L } |] ]

    withFile (Array.zeroCreate 16) (fun path ->
        for locations in invalidLocations do
            use file = File.OpenRead(path)
            Assert.Throws<ArgumentException>(fun () -> new FileTensorSource(file, locations) |> ignore) |> ignore)

[<Fact>]
let ``file source closes its file when disposed`` () =
    let locations = [| location "matrix" (Some TensorEncoding.F32) [ 1UL; 2UL ] 8 0L |]

    withFile (Array.zeroCreate 8) (fun path ->
        let file = File.OpenRead(path)
        let source = new FileTensorSource(file, locations) :> ITensorSource
        source.Dispose()

        Assert.False(file.CanRead)
        Assert.Throws<ObjectDisposedException>(fun () -> source.ReadPayload("matrix") |> ignore) |> ignore
        use exclusive = File.Open(path, FileMode.Open, FileAccess.ReadWrite, FileShare.None)
        Assert.True(exclusive.CanWrite))

[<Fact>]
let ``file source reports a file truncated after it was opened`` () =
    let locations = [| location "matrix" (Some TensorEncoding.F32) [ 2UL; 2UL ] 16 0L |]

    withFile (Array.zeroCreate 16) (fun path ->
        let file = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite)
        use source = new FileTensorSource(file, locations) :> ITensorSource

        do
            use writer = new FileStream(path, FileMode.Open, FileAccess.Write, FileShare.ReadWrite)
            writer.SetLength(10L)

        Assert.Throws<EndOfStreamException>(fun () -> source.ReadPayload("matrix") |> ignore) |> ignore)
