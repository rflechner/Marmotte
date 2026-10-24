namespace Marmotte.Cli

open System
open System.Globalization
open Marmotte.Gguf
open Marmotte.Cli.ConsoleDsl

module internal GgufReport =
    let private culture = CultureInfo.InvariantCulture

    let rec private formatValue depth = function
        | GgufValue.String value -> sprintf "\"%s\"" (previewText 120 value)
        | GgufValue.Array(elementType, values) ->
            let summary = sprintf "%A[%d]" elementType values.Length
            if depth = 0 || values.Length = 0 then summary
            else
                let preview =
                    values |> Array.truncate 3 |> Array.map (formatValue (depth - 1)) |> String.concat "; "
                let suffix = if values.Length > 3 then "; …" else ""
                sprintf "%s [%s%s]" summary preview suffix
        | value -> sprintf "%A" value

    let private writeSummary console fileName (fileSize: int64) (elapsed: TimeSpan) (model: GgufFile) =
        let metadata key =
            model.Metadata |> Map.tryFind key |> Option.map (formatValue 1) |> Option.defaultValue "not provided"
        let elementCount = model.Tensors |> Array.sumBy (fun tensor -> decimal tensor.ElementCount)

        table [ "Property"; "Value" ]
        |> hideHeaders
        |> rows [
            [ "File"; previewText 160 fileName ]
            [ "Name"; metadata "general.name" ]
            [ "Architecture"; metadata "general.architecture" ]
            [ "Quantization profile"; model.FileType |> Option.map string |> Option.defaultValue "not provided" ]
            [ "Format"; sprintf "GGUF v%d, %A" model.Version model.Endianness ]
            [ "File size"; formatBytes (decimal fileSize) ]
            [ "Metadata entries"; string model.Metadata.Count ]
            [ "Tensors"; string model.Tensors.Length ]
            [ "Tensor elements"; elementCount.ToString("N0", culture) ]
            [ "Alignment"; sprintf "%d bytes" model.Alignment ]
            [ "Tensor data offset"; sprintf "%d (0x%X)" model.DataOffset model.DataOffset ]
            [ "Index parsing"; elapsed.TotalMilliseconds.ToString("N1", culture) + " ms" ]
        ]
        |> writeSection console "GGUF summary"

    let private writeEncodings console (tensors: GgufTensorInfo array) =
        let groups = tensors |> Array.groupBy (fun tensor -> tensor.TensorType) |> Array.sortBy fst
        table [ "Encoding"; "Tensors"; "Elements"; "Bytes" ]
        |> rows (seq {
            for encoding, group in groups do
                let elements = group |> Array.sumBy (fun tensor -> decimal tensor.ElementCount)
                let sizes = group |> Array.choose (fun tensor -> tensor.ByteSize)
                let size =
                    if sizes.Length = group.Length then sizes |> Array.sumBy decimal |> formatBytes
                    else "unknown"
                yield [ string encoding; string group.Length; elements.ToString("N0", culture); size ]
        })
        |> writeSection console "Tensor encodings"

    let private writeMetadata console (metadata: Map<string, GgufValue>) =
        table [ "Key"; "Value" ]
        |> rows (seq {
            for KeyValue(key, value) in metadata do
                yield [ previewText 120 key; formatValue 2 value ]
        })
        |> writeSection console "Metadata (abbreviated values)"

    let private writeTensors console limit (tensors: GgufTensorInfo array) =
        let title = sprintf "Tensor preview (%d of %d, GGML dimension order)" (min limit tensors.Length) tensors.Length
        table [ "Name"; "Encoding"; "Dimensions"; "Bytes"; "File offset" ]
        |> rows (seq {
            for tensor in tensors |> Array.truncate limit do
                let shape =
                    if tensor.Dimensions.Length = 0 then "scalar"
                    else tensor.Dimensions |> Array.map string |> String.concat " × "
                let size = tensor.ByteSize |> Option.map (decimal >> formatBytes) |> Option.defaultValue "unknown"
                yield [ previewText 64 tensor.Name; string tensor.TensorType; shape; size; string tensor.AbsoluteOffset ]
        })
        |> writeSection console title

    /// Render the parsed index; dimensions retain GGML order and unknown byte sizes stay unknown.
    let write console fileName fileSize elapsed showMetadata tensorLimit (model: GgufFile) =
        writeSummary console fileName fileSize elapsed model
        if model.Tensors.Length > 0 then
            writeEncodings console model.Tensors
        if showMetadata then
            writeMetadata console model.Metadata
        if tensorLimit > 0 && model.Tensors.Length > 0 then
            writeTensors console tensorLimit model.Tensors
