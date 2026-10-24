module Marmotte.Cli.Program

open System
open System.Globalization
open System.Threading
open Spectre.Console.Cli

[<EntryPoint>]
let main args =
    use cancellation = new CancellationTokenSource()
    // The first Ctrl+C requests a clean cancellation. A second one is not intercepted and
    // terminates the process, for example if an I/O operation does not observe the token.
    let cancelHandler = ConsoleCancelEventHandler(fun _ event ->
        if not cancellation.IsCancellationRequested then
            event.Cancel <- true
            cancellation.Cancel())
    let app = CommandApp()
    app.Configure(fun config ->
        config.Settings.Culture <- CultureInfo.InvariantCulture
        config.SetApplicationName("marmotte") |> ignore
        config.CancellationExitCode(130) |> ignore
        config.SetExceptionHandler(fun error _ ->
            match error with
            | :? OperationCanceledException when cancellation.IsCancellationRequested ->
                Console.Error.WriteLine("Cancelled.")
                130
            | _ ->
                Console.Error.WriteLine("Error: " + ConsoleDsl.previewText 512 error.Message)
                1) |> ignore
        config.AddCommand<InfoCommand>("info")
            .WithDescription("Inspect a GGUF or Safetensors file without loading its weights.")
            .WithExample("info", "model.gguf")
            .WithExample("info", "model.safetensors")
            .WithExample("info", "model.gguf", "--metadata", "--tensor-limit", "20")
        |> ignore
        config.AddCommand<QuantizeCommand>("quantize")
            .WithDescription("Quantize F32/F16/BF16 GGUF weights into a new Q8_0 or Q4_0 file.")
            .WithExample("quantize", "model-f16.gguf", "model-q8_0.gguf")
            .WithExample("quantize", "model-f16.gguf", "model-q4_0.gguf", "--type", "Q4_0")
        |> ignore)
    Console.CancelKeyPress.AddHandler(cancelHandler)
    try
        app.Run(args, cancellation.Token)
    finally
        Console.CancelKeyPress.RemoveHandler(cancelHandler)
