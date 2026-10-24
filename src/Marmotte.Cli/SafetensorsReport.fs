namespace Marmotte.Cli

open System
open System.Globalization
open Marmotte.Safetensors
open Marmotte.Cli.ConsoleDsl

module internal SafetensorsReport =
    let private culture = CultureInfo.InvariantCulture

    let private formatShape (shape: uint64 array) =
        if shape.Length = 0 then "scalar"
        else shape |> Array.map string |> String.concat " × "

    let private writeSummary console fileName (fileSize: int64) (elapsed: TimeSpan) (model: SafetensorsFile) =
        let elementCount = model.Tensors |> Array.sumBy (fun tensor -> decimal tensor.ElementCount)

        table [ "Property"; "Value" ]
        |> hideHeaders
        |> rows [
            [ "File"; previewText 160 fileName ]
            [ "Format"; "Safetensors" ]
            [ "File size"; formatBytes (decimal fileSize) ]
            [ "JSON header"; formatBytes (decimal model.HeaderLength) ]
            [ "Metadata entries"; string model.Metadata.Count ]
            [ "Tensors"; string model.Tensors.Length ]
            [ "Tensor elements"; elementCount.ToString("N0", culture) ]
            [ "Tensor data offset"; sprintf "%d (0x%X)" model.DataOffset model.DataOffset ]
            [ "Tensor data size"; formatBytes (decimal model.DataLength) ]
            [ "Index parsing"; elapsed.TotalMilliseconds.ToString("N1", culture) + " ms" ]
        ]
        |> writeSection console "Safetensors summary"

    let private writeDtypes console (tensors: SafetensorsTensorInfo array) =
        let groups = tensors |> Array.groupBy (fun tensor -> tensor.Dtype) |> Array.sortBy fst

        table [ "Dtype"; "Tensors"; "Elements"; "Bytes" ]
        |> rows (seq {
            for dtype, group in groups do
                let elements = group |> Array.sumBy (fun tensor -> decimal tensor.ElementCount)
                let bytes = group |> Array.sumBy (fun tensor -> decimal tensor.ByteSize)
                yield [ dtype.WireName; string group.Length; elements.ToString("N0", culture); formatBytes bytes ]
        })
        |> writeSection console "Tensor dtypes"

    let private writeMetadata console (metadata: Map<string, string>) =
        table [ "Key"; "Value" ]
        |> rows (seq {
            for KeyValue(key, value) in metadata do
                yield [ previewText 120 key; sprintf "\"%s\"" (previewText 120 value) ]
        })
        |> writeSection console "Metadata (abbreviated values)"

    let private writeTensors console limit (tensors: SafetensorsTensorInfo array) =
        let shown = min limit tensors.Length
        let title = sprintf "Tensor preview (%d of %d, data offset order, row-major shapes)" shown tensors.Length

        table [ "Name"; "Dtype"; "Shape"; "Bytes"; "File offset" ]
        |> rows (seq {
            for tensor in tensors |> Array.truncate limit do
                yield [ previewText 96 tensor.Name
                        tensor.Dtype.WireName
                        formatShape tensor.Shape
                        formatBytes (decimal tensor.ByteSize)
                        string tensor.AbsoluteOffset ]
        })
        |> writeSection console title

    /// Render the parsed index. Tensors appear in data offset order; shapes keep row-major
    /// order, with the contiguous dimension last, unlike GGML's order in the GGUF report.
    let write console fileName fileSize elapsed showMetadata tensorLimit (model: SafetensorsFile) =
        writeSummary console fileName fileSize elapsed model
        if model.Tensors.Length > 0 then
            writeDtypes console model.Tensors
        if showMetadata then
            writeMetadata console model.Metadata
        if tensorLimit > 0 && model.Tensors.Length > 0 then
            writeTensors console tensorLimit model.Tensors
