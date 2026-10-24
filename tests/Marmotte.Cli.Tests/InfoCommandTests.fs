module Marmotte.Cli.Tests

open System
open System.IO
open System.Text
open Marmotte.Cli.TestSupport
open Xunit

// Encode a small GGUF directly so the CLI tests do not need installed models.
let private ggufBytes secondEncoding =
    use stream = new MemoryStream()
    use writer = new BinaryWriter(stream, Encoding.UTF8, true)
    let writeString (value: string) =
        let bytes = Encoding.UTF8.GetBytes value
        writer.Write(uint64 bytes.Length)
        writer.Write(bytes)
    let writeTextMetadata key value =
        writeString key
        writer.Write(8u)
        writeString value
    let writeTensor name (dimensions: uint64 array) (encoding: uint32) (offset: uint64) =
        writeString name
        writer.Write(uint32 dimensions.Length)
        for dimension in dimensions do writer.Write(dimension)
        writer.Write(encoding)
        writer.Write(offset)

    writer.Write(Encoding.ASCII.GetBytes "GGUF")
    writer.Write(3u)
    writer.Write(2UL)
    writer.Write(4UL)
    writeTextMetadata "general.name" "[red]demo[/]\u001b[31m"
    writeTextMetadata "general.architecture" "llama"
    writeString "general.file_type"
    writer.Write(4u)
    writer.Write(11u) // Q3_K_S file profile, independent of the tensor encodings.
    writeString "tokenizer.ggml.tokens"
    writer.Write(9u)
    writer.Write(8u)
    writer.Write(4UL)
    for token in [ "first[red]"; "second"; "third"; "hidden-token" ] do
        writeString token
    writeTensor "weight[0]" [| 2UL; 3UL |] 0u 0UL
    writeTensor "second_weight" [| 2UL |] secondEncoding 32UL
    while stream.Position % 32L <> 0L do writer.Write(0uy)
    writer.Write(Array.zeroCreate<byte> 36)
    stream.ToArray()

[<Fact>]
let ``help describes the file argument and inspection options`` () =
    let result = run [ "info"; "--help" ]
    Assert.Equal(0, result.ExitCode)
    Assert.Contains("<FILE>", result.Output)
    Assert.Contains("USAGE:", result.Output)
    Assert.Contains("--metadata", result.Output)
    Assert.Contains("--tensor-limit", result.Output)
    Assert.Empty(result.Error)

[<Fact>]
let ``inspects a file with spaces and brackets without a local environment setting`` () =
    use model = new ModelFile(ggufBytes 1u)
    let result = run [ "info"; model.Path ]
    Assert.True(result.ExitCode = 0, result.Error)
    Assert.Contains("[sample] model.gguf", result.Output)
    Assert.Contains("[red]demo[/]\\u001b[31m", result.Output)
    Assert.Contains("llama", result.Output)
    Assert.Contains("Q3_K_S", result.Output)
    Assert.Contains("GGUF v3, LittleEndian", result.Output)
    Assert.Contains("F32", result.Output)
    Assert.Contains("F16", result.Output)
    Assert.Contains("2 × 3", result.Output)
    Assert.Contains("second_weight", result.Output)
    Assert.DoesNotContain(model.Path, result.Output)
    Assert.DoesNotContain("tokenizer.ggml.tokens", result.Output)
    Assert.DoesNotContain("\u001b", result.Output, StringComparison.Ordinal)
    Assert.Empty(result.Error)

[<Fact>]
let ``metadata values are literal and abbreviated and the tensor limit is respected`` () =
    use model = new ModelFile(ggufBytes 1u)
    let result = run [ "info"; model.Path; "--metadata"; "--tensor-limit"; "1" ]
    Assert.True(result.ExitCode = 0, result.Error)
    Assert.Contains("tokenizer.ggml.tokens", result.Output)
    Assert.Contains("String[4]", result.Output)
    Assert.Contains("first[red]", result.Output)
    Assert.DoesNotContain("hidden-token", result.Output)
    Assert.Contains("weight[0]", result.Output)
    Assert.DoesNotContain("second_weight", result.Output)

[<Fact>]
let ``zero tensor limit hides the preview but keeps aggregate information`` () =
    use model = new ModelFile(ggufBytes 1u)
    let result = run [ "info"; model.Path; "--tensor-limit"; "0" ]
    Assert.True(result.ExitCode = 0, result.Error)
    Assert.Contains("Tensor encodings", result.Output)
    Assert.DoesNotContain("Tensor preview", result.Output)
    Assert.DoesNotContain("weight[0]", result.Output)

[<Fact>]
let ``unknown encodings have unknown sizes and preserve their identifier`` () =
    use model = new ModelFile(ggufBytes 999u)
    let result = run [ "info"; model.Path ]
    Assert.True(result.ExitCode = 0, result.Error)
    Assert.Contains("999", result.Output)
    Assert.Contains("unknown", result.Output)

[<Theory>]
[<InlineData("-1")>]
[<InlineData("text")>]
let ``invalid tensor limits fail before attempting to read a file`` limit =
    let result = run [ "info"; "missing.gguf"; "--tensor-limit"; limit ]
    Assert.Equal(1, result.ExitCode)
    Assert.Contains("Error:", result.Error)
    Assert.DoesNotContain("Could not find", result.Error)
    Assert.Empty(result.Output)

[<Fact>]
let ``a missing file argument produces an error on stderr`` () =
    let result = run [ "info" ]
    Assert.Equal(1, result.ExitCode)
    Assert.Contains("Error:", result.Error)
    Assert.Empty(result.Output)

[<Fact>]
let ``a missing file produces an error without a stack trace`` () =
    use model = new ModelFile(ggufBytes 1u)
    File.Delete(model.Path)
    let result = run [ "info"; model.Path ]
    Assert.Equal(1, result.ExitCode)
    Assert.Contains("Error:", result.Error)
    Assert.DoesNotContain(" at ", result.Error)
    Assert.Empty(result.Output)

[<Theory>]
[<InlineData(0)>]
[<InlineData(12)>]
let ``truncated files fail without printing a partial report`` length =
    use model = new ModelFile(ggufBytes 1u |> Array.take length)
    let result = run [ "info"; model.Path ]
    Assert.Equal(1, result.ExitCode)
    Assert.Contains("Error:", result.Error)
    Assert.Contains("byte", result.Error)
    Assert.Empty(result.Output)

[<Fact>]
let ``a gguf extension does not make an invalid file acceptable`` () =
    use model = new ModelFile(Array.create 32 0uy)
    let result = run [ "info"; model.Path ]
    Assert.Equal(1, result.ExitCode)
    Assert.Contains("Error:", result.Error)
    Assert.Empty(result.Output)
