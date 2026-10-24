module Marmotte.Cli.QuantizeCommandTests

open System
open System.IO
open System.Text
open Marmotte.Cli.TestSupport
open Marmotte.Gguf
open Xunit

let private ggufBytes encoding =
    use stream = new MemoryStream()
    use writer = new BinaryWriter(stream, Encoding.UTF8, true)
    let writeString (value: string) =
        let bytes = Encoding.UTF8.GetBytes value
        writer.Write(uint64 bytes.Length)
        writer.Write(bytes)
    writer.Write(Encoding.ASCII.GetBytes "GGUF")
    writer.Write(3u)
    writer.Write(1UL)
    writer.Write(0UL)
    writeString "layer.weight"
    writer.Write(2u)
    writer.Write(32UL)
    writer.Write(2UL)
    writer.Write(encoding: uint32)
    writer.Write(0UL)
    while stream.Position % 32L <> 0L do writer.Write(0uy)
    if encoding = 0u then
        for index in 0 .. 63 do writer.Write(float32 (index % 16 - 8))
    else
        writer.Write(Array.zeroCreate<byte> 36) // Two Q4_0 blocks.
    stream.ToArray()

[<Fact>]
let ``quantize help advertises input output supported encodings and default`` () =
    let result = run [ "quantize"; "--help" ]
    Assert.Equal(0, result.ExitCode)
    for expected in [ "<INPUT>"; "<OUTPUT>"; "--type"; "Q8_0"; "Q4_0"; "F32"; "F16"; "BF16" ] do
        Assert.Contains(expected, result.Output)
    Assert.Empty(result.Error)

[<Theory>]
[<InlineData("")>]
[<InlineData("q4_0")>]
let ``quantizes to a separate file that info can inspect`` format =
    let bytes = ggufBytes 0u
    use model = new ModelFile(bytes)
    let arguments = [ "quantize"; model.Path; model.OutputPath ]
    let arguments = if format = "" then arguments else arguments @ [ "--type"; format ]
    let result = run arguments
    Assert.True(result.ExitCode = 0, result.Error)
    Assert.Empty(result.Error)
    Assert.Contains("Quantization complete", result.Output)
    Assert.Contains("[quantized] model.gguf", result.Output)
    Assert.Contains("Quantized tensors", result.Output)
    Assert.Contains("Preserved tensors", result.Output)
    Assert.DoesNotContain(model.OutputPath, result.Output)
    Assert.DoesNotContain("\u001b", result.Output, StringComparison.Ordinal)
    Assert.Equal<byte>(bytes, File.ReadAllBytes model.Path)
    let expected = if format = "" then GgufFileType.Q8_0 else GgufFileType.Q4_0
    Assert.Equal(Some expected, (Gguf.readFile model.OutputPath).FileType)
    let inspection = run [ "info"; model.OutputPath ]
    Assert.True(inspection.ExitCode = 0, inspection.Error)
    Assert.Contains(string expected, inspection.Output)
    Assert.Equal(2, Directory.GetFiles(model.Directory).Length)

[<Theory>]
[<InlineData("Q4_K_M")>]
[<InlineData("Q3_K_S")>]
[<InlineData("F16")>]
[<InlineData("7")>]
let ``unsupported target formats fail before opening input`` format =
    let result = run [ "quantize"; "missing.gguf"; "output.gguf"; "--type"; format ]
    Assert.Equal(1, result.ExitCode)
    Assert.Contains("Q8_0 or Q4_0", result.Error)
    Assert.DoesNotContain("Could not find", result.Error)
    Assert.DoesNotContain("\n   at ", result.Error)

[<Fact>]
let ``requires an output argument`` () =
    let result = run [ "quantize"; "missing.gguf" ]
    Assert.Equal(1, result.ExitCode)
    Assert.Contains("Error:", result.Error)
    Assert.DoesNotContain("Could not find", result.Error)

[<Fact>]
let ``refuses already quantized input without creating output`` () =
    use model = new ModelFile(ggufBytes 2u)
    let result = run [ "quantize"; model.Path; model.OutputPath ]
    Assert.Equal(1, result.ExitCode)
    Assert.Contains("requantization is not implemented", result.Error)
    Assert.DoesNotContain("Quantization complete", result.Output)
    Assert.Single(Directory.GetFiles model.Directory) |> ignore

[<Fact>]
let ``refuses overwriting source or existing output`` () =
    let bytes = ggufBytes 0u
    use model = new ModelFile(bytes)
    let samePath = run [ "quantize"; model.Path; model.Path ]
    Assert.Equal(1, samePath.ExitCode)
    Assert.Contains("different files", samePath.Error)
    let originalOutput = [| 10uy; 20uy |]
    File.WriteAllBytes(model.OutputPath, originalOutput)
    let existing = run [ "quantize"; model.Path; model.OutputPath ]
    Assert.Equal(1, existing.ExitCode)
    Assert.Contains("already exists", existing.Error)
    Assert.Equal<byte>(bytes, File.ReadAllBytes model.Path)
    Assert.Equal<byte>(originalOutput, File.ReadAllBytes model.OutputPath)

[<Fact>]
let ``malformed input reports an error without creating output`` () =
    use model = new ModelFile(Array.zeroCreate 24)
    let result = run [ "quantize"; model.Path; model.OutputPath ]
    Assert.Equal(1, result.ExitCode)
    Assert.Contains("Error:", result.Error)
    Assert.DoesNotContain("\n   at ", result.Error)
    Assert.Single(Directory.GetFiles model.Directory) |> ignore
