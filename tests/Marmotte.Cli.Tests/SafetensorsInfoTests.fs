module Marmotte.Cli.SafetensorsInfoTests

open System
open System.Text
open Marmotte.Cli.TestSupport
open Xunit

// Encode a small Safetensors file directly so the CLI tests do not need installed models.
let private safetensorsBytes (header: string) (dataLength: int) =
    let headerBytes = Encoding.UTF8.GetBytes header
    let length = Array.init 8 (fun index -> byte (uint64 headerBytes.Length >>> (index * 8)))
    Array.concat [ length; headerBytes; Array.zeroCreate dataLength ]

// JSON order differs from data order, and the metadata contains markup and an escaped ESC.
let private sampleHeader =
    """{"lm_head.bias":{"dtype":"F32","shape":[2],"data_offsets":[12,20]},"__metadata__":{"format":"pt","note":"[red]demo[/]\u001b[31m"},"model.embed[0].weight":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,12]}}"""

let private sampleModel () =
    new ModelFile(safetensorsBytes sampleHeader 20, "[sample] model.safetensors")

[<Fact>]
let ``inspects a Safetensors file in data offset order`` () =
    use model = sampleModel ()
    let result = run [ "info"; model.Path ]

    Assert.True(result.ExitCode = 0, result.Error)
    Assert.Contains("Safetensors summary", result.Output)
    Assert.Contains("[sample] model.safetensors", result.Output)
    Assert.Contains("BF16", result.Output)
    Assert.Contains("F32", result.Output)
    Assert.Contains("2 × 3", result.Output)
    Assert.Contains("model.embed[0].weight", result.Output)
    Assert.Contains("lm_head.bias", result.Output)
    Assert.True(result.Output.IndexOf("model.embed[0].weight") < result.Output.IndexOf("lm_head.bias"))
    Assert.DoesNotContain("demo", result.Output)
    Assert.DoesNotContain(model.Path, result.Output)
    Assert.Empty(result.Error)

[<Fact>]
let ``metadata values are literal and the tensor limit is respected`` () =
    use model = sampleModel ()
    let result = run [ "info"; model.Path; "--metadata"; "--tensor-limit"; "1" ]

    Assert.True(result.ExitCode = 0, result.Error)
    Assert.Contains("\"pt\"", result.Output)
    Assert.Contains("[red]demo[/]\\u001b[31m", result.Output)
    Assert.DoesNotContain("\u001b", result.Output, StringComparison.Ordinal)
    Assert.Contains("model.embed[0].weight", result.Output)
    Assert.DoesNotContain("lm_head.bias", result.Output)

[<Fact>]
let ``detects the format from content rather than the extension`` () =
    use model = new ModelFile(safetensorsBytes sampleHeader 20, "[sample] model.gguf")
    let result = run [ "info"; model.Path; "--tensor-limit"; "0" ]

    Assert.True(result.ExitCode = 0, result.Error)
    Assert.Contains("Safetensors summary", result.Output)
    Assert.Contains("Tensor dtypes", result.Output)
    Assert.DoesNotContain("Tensor preview", result.Output)

[<Fact>]
let ``invalid Safetensors files fail without printing a partial report`` () =
    // The last two data bytes are missing.
    use model = new ModelFile(safetensorsBytes sampleHeader 18, "[sample] model.safetensors")
    let result = run [ "info"; model.Path ]

    Assert.Equal(1, result.ExitCode)
    Assert.Contains("Invalid Safetensors header", result.Error)
    Assert.DoesNotContain("\n   at ", result.Error)
    Assert.Empty(result.Output)

[<Fact>]
let ``unrecognized files report both expected signatures`` () =
    use model = new ModelFile(Encoding.ASCII.GetBytes "not a model file", "[sample] model.safetensors")
    let result = run [ "info"; model.Path ]

    Assert.Equal(1, result.ExitCode)
    Assert.Contains("Unrecognized file format", result.Error)
    Assert.Empty(result.Output)
