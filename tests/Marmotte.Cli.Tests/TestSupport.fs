module internal Marmotte.Cli.TestSupport

open System
open System.Diagnostics
open System.IO
open Marmotte.Cli

type ModelFile(bytes: byte array) =
    let directory = Directory.CreateTempSubdirectory("marmotte-cli-tests-")
    let path = Path.Combine(directory.FullName, "[sample] model.gguf")
    do File.WriteAllBytes(path, bytes)
    member _.Path = path
    member _.Directory = directory.FullName
    member _.OutputPath = Path.Combine(directory.FullName, "[quantized] model.gguf")
    interface IDisposable with
        member _.Dispose() = directory.Delete(true)

type CommandResult =
    { ExitCode: int
      Output: string
      Error: string }

let run arguments =
    let startInfo = ProcessStartInfo("dotnet")
    startInfo.UseShellExecute <- false
    startInfo.RedirectStandardOutput <- true
    startInfo.RedirectStandardError <- true
    startInfo.ArgumentList.Add(typeof<InfoCommand>.Assembly.Location)
    for argument in arguments do startInfo.ArgumentList.Add(argument)
    startInfo.Environment["NO_COLOR"] <- "1"
    startInfo.Environment["TERM"] <- "dumb"
    startInfo.Environment.Remove("LM_STUDIO_MODELS_FOLDER") |> ignore
    use child = Process.Start(startInfo)
    let output = child.StandardOutput.ReadToEndAsync()
    let error = child.StandardError.ReadToEndAsync()
    if not (child.WaitForExit(15_000)) then
        child.Kill(true)
        failwith "CLI did not exit within 15 seconds."
    { ExitCode = child.ExitCode
      Output = output.GetAwaiter().GetResult()
      Error = error.GetAwaiter().GetResult() }
