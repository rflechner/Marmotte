module Marmotte.Cli.Program

open System
open System.Globalization
open Spectre.Console.Cli

[<EntryPoint>]
let main args =
    let app = CommandApp()
    app.Configure(fun config ->
        config.Settings.Culture <- CultureInfo.InvariantCulture
        config.SetApplicationName("marmotte") |> ignore
        config.SetExceptionHandler(fun error _ ->
            Console.Error.WriteLine("Error: " + ConsoleDsl.previewText 512 error.Message)
            1) |> ignore
        config.AddCommand<InfoCommand>("info")
            .WithDescription("Inspect a GGUF file without loading its weights.")
            .WithExample("info", "model.gguf")
            .WithExample("info", "model.gguf", "--metadata", "--tensor-limit", "20")
        |> ignore)
    app.Run(args)
