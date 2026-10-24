#load "../Marmotte.Core/Helpers.fs"
#load "Types.fs"
#load "Gguf.fs"

open System
open System.Diagnostics
open System.Globalization
open System.IO
open Marmotte.Gguf
open Marmotte.Core.Helpers

let modelsFolder =
    match Environment.GetEnvironmentVariable "LM_STUDIO_MODELS_FOLDER" with
    | folder when String.IsNullOrWhiteSpace folder ->
        failwith "LM_STUDIO_MODELS_FOLDER is not set; see README.md, section F# exploration script."
    | folder -> folder

let filePath =
    modelsFolder />
        "lmstudio-community" />
        "Ministral-3-3B-Instruct-2512-GGUF" />
        "Ministral-3-3B-Instruct-2512-Q4_K_M.gguf"

if not (File.Exists filePath) then failwithf "File not found: %s" filePath


let maxDisplayedTensors = 12
let maxArrayPreview = 3
let maxStringLength = 120
let culture = CultureInfo.GetCultureInfo("en-US")

let formatBytes (bytes: uint64) =
    let units = [| "B"; "KiB"; "MiB"; "GiB"; "TiB" |]
    let mutable size = float bytes
    let mutable index = 0
    while size >= 1024.0 && index < units.Length - 1 do
        size <- size / 1024.0
        index <- index + 1
    sprintf "%s %s" (size.ToString("N2", culture)) units[index]

// Large strings and arrays (especially the vocabulary) are abbreviated for display.
// Their complete contents remain available in model.Metadata.
let rec formatValue depth = function
    | GgufValue.String text ->
        let text = text.Replace("\r", "\\r").Replace("\n", "\\n").Replace("\t", "\\t")
        let preview = if text.Length > maxStringLength then text[..maxStringLength - 1] + "…" else text
        sprintf "%A" preview
    | GgufValue.Array(elementType, values) ->
        let summary = sprintf "%A[%d]" elementType values.Length
        if depth = 0 then summary
        else
            let preview = values |> Array.truncate maxArrayPreview |> Array.map (formatValue (depth - 1)) |> String.concat "; "
            let suffix = if values.Length > maxArrayPreview then "; …" else ""
            sprintf "%s [%s%s]" summary preview suffix
    | value -> sprintf "%A" value

printfn "Reading %s" (Path.GetFileName filePath)
let timer = Stopwatch.StartNew()
let model = Gguf.readFile filePath
timer.Stop()

let parameterCount = model.Tensors |> Array.sumBy (fun tensor -> decimal tensor.ElementCount)
let printMetadata label key =
    let value = model.Metadata |> Map.tryFind key |> Option.map (formatValue 1) |> Option.defaultValue "not provided"
    printfn "%-24s %s" label value

printfn "\nSummary"
printMetadata "Name" "general.name"
printMetadata "Architecture" "general.architecture"
printfn "%-24s %s" "Quantization profile" (model.FileType |> Option.map string |> Option.defaultValue "not provided")
printfn "%-24s v%d, %A" "Format" model.Version model.Endianness
printfn "%-24s %s" "File size" (formatBytes (uint64 (FileInfo(filePath).Length)))
printfn "%-24s %d" "Metadata entries" model.Metadata.Count
printfn "%-24s %d" "Tensors" model.Tensors.Length
printfn "%-24s %s" "Tensor elements" (parameterCount.ToString("N0", culture))
printfn "%-24s %d bytes" "Alignment" model.Alignment
printfn "%-24s %d (0x%X)" "Tensor data offset" model.DataOffset model.DataOffset
printfn "%-24s %s ms" "Index parsing" (timer.Elapsed.TotalMilliseconds.ToString("N1", culture))

printfn "\nTensor distribution by encoding"
printfn "%-12s %8s %20s %16s" "Encoding" "Tensors" "Elements" "Weights"
for encoding, tensors in model.Tensors |> Array.groupBy (fun tensor -> tensor.TensorType) |> Array.sortBy fst do
    let elements = tensors |> Array.sumBy (fun tensor -> decimal tensor.ElementCount)
    let sizes = tensors |> Array.choose (fun tensor -> tensor.ByteSize)
    let size = if sizes.Length = tensors.Length then formatBytes (Array.sum sizes) else "unknown"
    printfn "%-12s %8d %20s %16s" (string encoding) tensors.Length (elements.ToString("N0", culture)) size

printfn "\nMetadata (previews limited to %d elements per array)" maxArrayPreview
for KeyValue(key, value) in model.Metadata do
    printfn "  %-48s %s" key (formatValue 2 value)

printfn "\nTensor preview (%d of %d, dimensions in GGML order)" (min maxDisplayedTensors model.Tensors.Length) model.Tensors.Length
printfn "%-42s %-10s %-24s %14s %16s" "Name" "Encoding" "Dimensions" "Size" "Absolute offset"
for tensor in model.Tensors |> Array.truncate maxDisplayedTensors do
    let shape = if tensor.Dimensions.Length = 0 then "scalar" else tensor.Dimensions |> Array.map string |> String.concat " × "
    let size = tensor.ByteSize |> Option.map formatBytes |> Option.defaultValue "unknown"
    printfn "%-42s %-10s %-24s %14s %16d" tensor.Name (string tensor.TensorType) shape size tensor.AbsoluteOffset
