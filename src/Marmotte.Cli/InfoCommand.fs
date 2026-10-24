namespace Marmotte.Cli

open System
open System.ComponentModel
open System.Diagnostics
open System.IO
open Spectre.Console
open Spectre.Console.Cli
open Marmotte.Gguf

/// Arguments for inspecting the index of a local GGUF file.
type InfoSettings() =
    inherit CommandSettings()

    [<CommandArgument(0, "<FILE>")>]
    [<Description("Path to the GGUF file to inspect.")>]
    member val FilePath = "" with get, set

    [<CommandOption("--metadata")>]
    [<Description("Show all metadata entries, with abbreviated strings and arrays.")>]
    member val ShowMetadata = false with get, set

    [<CommandOption("--tensor-limit <COUNT>")>]
    [<Description("Maximum number of tensor descriptors to display; 0 hides the preview.")>]
    [<DefaultValue(12)>]
    member val TensorLimit = 12 with get, set

    override this.Validate() =
        if String.IsNullOrWhiteSpace this.FilePath then
            ValidationResult.Error("A GGUF file path is required.")
        elif this.TensorLimit < 0 then
            ValidationResult.Error("--tensor-limit must be zero or greater.")
        else
            ValidationResult.Success()

/// Inspect metadata and tensor descriptors without reading the model's weights.
type InfoCommand() =
    inherit Command<InfoSettings>()

    override _.Execute(_, settings, _) =
        use stream = File.OpenRead settings.FilePath
        let timer = Stopwatch.StartNew()
        let model = Gguf.read stream
        timer.Stop()

        GgufReport.write AnsiConsole.Console (Path.GetFileName settings.FilePath)
            stream.Length timer.Elapsed settings.ShowMetadata settings.TensorLimit model
        0
