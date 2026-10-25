namespace Marmotte.Gguf

open System
open System.IO
open Marmotte.Core

/// Maps GGUF indexes to Marmotte.Core's format-neutral tensor source.
[<RequireQualifiedAccess>]
module GgufTensorSource =
    /// Runtime encoding of a GGML type, or None when Marmotte does not represent it.
    let tryEncoding (tensorType: GgmlType) : TensorEncoding option =
        match tensorType with
        | GgmlType.F32 -> Some TensorEncoding.F32
        | GgmlType.F16 -> Some TensorEncoding.F16
        | GgmlType.BF16 -> Some TensorEncoding.BF16
        | GgmlType.Q4_0 -> Some TensorEncoding.Q4_0
        | GgmlType.Q4_1 -> Some TensorEncoding.Q4_1
        | GgmlType.Q5_0 -> Some TensorEncoding.Q5_0
        | GgmlType.Q5_1 -> Some TensorEncoding.Q5_1
        | GgmlType.Q8_0 -> Some TensorEncoding.Q8_0
        | GgmlType.IQ4_NL -> Some TensorEncoding.IQ4_NL
        | GgmlType.Q2_K -> Some TensorEncoding.Q2_K
        | GgmlType.Q3_K -> Some TensorEncoding.Q3_K
        | GgmlType.Q4_K -> Some TensorEncoding.Q4_K
        | GgmlType.Q5_K -> Some TensorEncoding.Q5_K
        | GgmlType.Q6_K -> Some TensorEncoding.Q6_K
        | _ -> None

    /// Maps a GGUF descriptor. Dimensions are reversed from GGML order into row-major order:
    /// both describe the same bytes, with the contiguous dimension moving from first to last.
    /// Unknown GGML types keep their numeric identifier in SourceEncoding.
    let descriptor (tensor: GgufTensorInfo) : TensorDescriptor =
        { Name = tensor.Name
          Encoding = tryEncoding tensor.TensorType
          SourceEncoding = string tensor.TensorType
          Shape = Array.rev tensor.Dimensions
          ElementCount = tensor.ElementCount
          ByteSize = tensor.ByteSize }

    let private location (tensor: GgufTensorInfo) : FileTensorLocation =
        { Descriptor = descriptor tensor
          AbsoluteOffset = tensor.AbsoluteOffset }

    // Payloads are served as little-endian bytes, and a split file holds only part of a model.
    let private validateForRuntime (model: GgufFile) =
        if model.Endianness <> GgufEndianness.LittleEndian then
            raise (InvalidDataException("Big-endian GGUF payloads are not supported; use a little-endian file."))

        let isSplitKey (key: string) _ = key.StartsWith("split.", StringComparison.Ordinal)
        if model.Metadata |> Map.exists isSplitKey then
            raise (InvalidDataException("Split GGUF files are not supported; use a complete, single-file model."))

    /// Opens a complete little-endian GGUF file read-only, reads its index with the given limits
    /// and keeps the file open until the source is disposed. Payloads are copied on request; see
    /// FileTensorSource. Malformed, big-endian and split files raise InvalidDataException and
    /// I/O errors propagate; the file is closed when opening fails.
    let openFileWithOptions (options: GgufReadOptions) (path: string) : ITensorSource =
        let file = File.OpenRead(path)
        try
            let model = Gguf.readWithOptions options file
            validateForRuntime model
            new FileTensorSource(file, Array.map location model.Tensors) :> ITensorSource
        with _ ->
            file.Dispose()
            reraise ()

    /// Opens a file with the default index limits.
    let openFile path = openFileWithOptions GgufReadOptions.Default path
