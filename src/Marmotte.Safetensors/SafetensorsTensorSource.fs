namespace Marmotte.Safetensors

open System.IO
open Marmotte.Core

/// Maps Safetensors indexes to Marmotte.Core's format-neutral tensor source.
[<RequireQualifiedAccess>]
module SafetensorsTensorSource =
    /// Runtime encoding of a dtype, or None when Marmotte does not represent it.
    let tryEncoding (dtype: SafetensorsDtype) : TensorEncoding option =
        match dtype with
        | SafetensorsDtype.F32 -> Some TensorEncoding.F32
        | SafetensorsDtype.F16 -> Some TensorEncoding.F16
        | SafetensorsDtype.BF16 -> Some TensorEncoding.BF16
        | _ -> None

    /// Maps a Safetensors descriptor. The shape is already row-major; it is copied so that the
    /// descriptor does not share its array with the index.
    let descriptor (tensor: SafetensorsTensorInfo) : TensorDescriptor =
        { Name = tensor.Name
          Encoding = tryEncoding tensor.Dtype
          SourceEncoding = tensor.Dtype.WireName
          Shape = Array.copy tensor.Shape
          ElementCount = tensor.ElementCount
          ByteSize = Some tensor.ByteSize }

    let private location (tensor: SafetensorsTensorInfo) : FileTensorLocation =
        { Descriptor = descriptor tensor
          AbsoluteOffset = tensor.AbsoluteOffset }

    /// Opens one Safetensors file read-only, reads its index with the given limits and keeps the
    /// file open until the source is disposed. A shard of a sharded model exposes only its own
    /// tensors. Payloads are copied on request; see FileTensorSource. Malformed files raise
    /// InvalidDataException and I/O errors propagate; the file is closed when opening fails.
    let openFileWithOptions (options: SafetensorsReadOptions) (path: string) : ITensorSource =
        let file = File.OpenRead(path)
        try
            let model = Safetensors.readWithOptions options file
            new FileTensorSource(file, Array.map location model.Tensors) :> ITensorSource
        with _ ->
            file.Dispose()
            reraise ()

    /// Opens a file with the default index limits.
    let openFile path = openFileWithOptions SafetensorsReadOptions.Default path
