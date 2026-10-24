namespace Marmotte.Cli

open System
open System.ComponentModel
open System.Diagnostics
open System.Globalization
open System.IO
open Marmotte.Gguf
open Spectre.Console
open Spectre.Console.Cli

/// Arguments for native quantization into a new GGUF file.
type QuantizeSettings() =
    inherit CommandSettings()

    [<CommandArgument(0, "<INPUT>")>]
    [<Description("Complete little-endian GGUF containing F32, F16 or BF16 weights.")>]
    member val InputPath = "" with get, set

    [<CommandArgument(1, "<OUTPUT>")>]
    [<Description("New output GGUF path; existing files are never overwritten.")>]
    member val OutputPath = "" with get, set

    [<CommandOption("--type <FORMAT>")>]
    [<Description("Target encoding: Q8_0 or Q4_0. Requantization and K-quants are not supported.")>]
    [<DefaultValue("Q8_0")>]
    member val Format = "Q8_0" with get, set

    member this.TargetFormat =
        match this.Format.ToUpperInvariant() with
        | "Q4_0" -> GgufQuantizationFormat.Q4_0
        | "Q8_0" -> GgufQuantizationFormat.Q8_0
        | _ -> invalidArg "format" "--type must be Q8_0 or Q4_0."

    override this.Validate() =
        if String.IsNullOrWhiteSpace this.InputPath || String.IsNullOrWhiteSpace this.OutputPath then
            ValidationResult.Error("Both input and output GGUF paths are required.")
        elif not (String.Equals(this.Format, "Q4_0", StringComparison.OrdinalIgnoreCase)
                  || String.Equals(this.Format, "Q8_0", StringComparison.OrdinalIgnoreCase)) then
            ValidationResult.Error("--type must be Q8_0 or Q4_0; K-quants are not implemented.")
        else
            ValidationResult.Success()

/// Convert floating-point tensors with bounded payload memory and publish a new GGUF on success.
type QuantizeCommand() =
    inherit Command<QuantizeSettings>()

    override _.Execute(_, settings, cancellationToken) =
        let timer = Stopwatch.StartNew()
        let result =
            AnsiConsole.Status().Start("Quantizing GGUF weights...", fun _ ->
                GgufQuantizer.quantizeFileWithCancellation settings.TargetFormat
                    settings.InputPath settings.OutputPath cancellationToken)
        timer.Stop()

        ConsoleDsl.table [ "Property"; "Value" ]
        |> ConsoleDsl.hideHeaders
        |> ConsoleDsl.rows [
            [ "Output"; ConsoleDsl.previewText 160 (Path.GetFileName settings.OutputPath) ]
            [ "Encoding"; string result.Format ]
            [ "Quantized tensors"; string result.QuantizedTensors ]
            [ "Preserved tensors"; string result.PreservedTensors ]
            [ "Input size"; ConsoleDsl.formatBytes (decimal result.InputBytes) ]
            [ "Output size"; ConsoleDsl.formatBytes (decimal result.OutputBytes) ]
            [ "Elapsed"; timer.Elapsed.TotalSeconds.ToString("N2", CultureInfo.InvariantCulture) + " s" ]
        ]
        |> ConsoleDsl.writeSection AnsiConsole.Console "Quantization complete"
        0
