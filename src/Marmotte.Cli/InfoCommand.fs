namespace Marmotte.Cli

open System
open System.ComponentModel
open System.Diagnostics
open System.IO
open Spectre.Console
open Spectre.Console.Cli
open Marmotte.Gguf
open Marmotte.Safetensors

/// Model file formats recognized from their leading bytes, never from the file extension.
[<RequireQualifiedAccess>]
type internal ModelFileFormat =
    | Gguf
    | Safetensors

module internal ModelFileFormat =
    /// GGUF starts with its ASCII magic. Safetensors starts with an 8-byte header length,
    /// followed by the '{' that opens its JSON header. The stream position is restored.
    let detect (stream: Stream) =
        let start = stream.Position
        let prefix = Array.zeroCreate<byte> 9
        let count = stream.ReadAtLeast(prefix, prefix.Length, throwOnEndOfStream = false)
        stream.Position <- start

        if count >= 4 && prefix[0..3] = "GGUF"B then
            ModelFileFormat.Gguf
        elif count = prefix.Length && prefix[8] = byte '{' then
            ModelFileFormat.Safetensors
        else
            raise (InvalidDataException("Unrecognized file format: expected the GGUF magic at byte 0 or a Safetensors JSON header at byte 8."))

/// Arguments for inspecting the index of a local GGUF or Safetensors file.
type InfoSettings() =
    inherit CommandSettings()

    [<CommandArgument(0, "<FILE>")>]
    [<Description("Path to the GGUF or Safetensors file to inspect; the format is detected from its content.")>]
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
            ValidationResult.Error("A model file path is required.")
        elif this.TensorLimit < 0 then
            ValidationResult.Error("--tensor-limit must be zero or greater.")
        else
            ValidationResult.Success()

/// Inspect metadata and tensor descriptors without reading the model's weights.
type InfoCommand() =
    inherit Command<InfoSettings>()

    override _.Execute(_, settings, _) =
        use stream = File.OpenRead settings.FilePath
        let fileName = Path.GetFileName settings.FilePath

        // Only index parsing is timed, not format detection or rendering.
        let timedRead read =
            let timer = Stopwatch.StartNew()
            let model = read stream
            timer.Stop()
            model, timer.Elapsed

        match ModelFileFormat.detect stream with
        | ModelFileFormat.Gguf ->
            let model, elapsed = timedRead Gguf.read
            GgufReport.write AnsiConsole.Console fileName
                stream.Length elapsed settings.ShowMetadata settings.TensorLimit model
        | ModelFileFormat.Safetensors ->
            let model, elapsed = timedRead Safetensors.read
            SafetensorsReport.write AnsiConsole.Console fileName
                stream.Length elapsed settings.ShowMetadata settings.TensorLimit model
        0
