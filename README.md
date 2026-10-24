# Marmotte

F# libraries for local inference, targeting .NET 10.

## GGUF reader

`Marmotte.Gguf` reads GGUF v2 (little-endian) and v3 files
(little-endian or automatically detected big-endian). The reader exposes typed
metadata and the tensor index without loading or dequantizing the weights.

```fsharp
open Marmotte.Gguf

let model = Gguf.readFile "model.gguf"

match model.Metadata |> Map.tryFind "general.name" with
| Some (GgufValue.String name) -> printfn "Model: %s" name
| _ -> ()

for tensor in model.Tensors do
    printfn "%s: %A, %A, position %d"
        tensor.Name tensor.Dimensions tensor.TensorType tensor.AbsoluteOffset
```

`Gguf.read stream` accepts a readable, seekable stream, starting at its current
position. The stream remains open and is positioned at `DataOffset` after
reading. `AbsoluteOffset` and `DataOffset` are absolute positions in that
stream; `Offset` is relative to the start of the tensor data section. Dimensions
are in GGML order, with the contiguous dimension first.

`model.FileType` exposes the file's quantization profile from `general.file_type`
through `GgufFileType`: `Q3_K_S`, `Q3_K_M`, `Q3_K_L`, `Q4_K_M`, etc. These profiles
can combine several tensor encodings. `tensor.TensorType` exposes each tensor's
actual encoding through `GgmlType`, such as `Q3_K`, `Q6_K`, or `F32`. The two
enumerations have distinct identifiers: the value 12 means `GgufFileType.Q3_K_M`
for the file and `GgmlType.Q4_K` for a tensor.

```fsharp
match model.FileType with
| Some GgufFileType.Q3_K_S -> printfn "Q3_K Small profile"
| Some profile -> printfn "Profile: %A" profile
| None -> printfn "Profile not provided as uint32"
```

The profile is `None` if this metadata is missing or has a type other than
`uint32`; it is not inferred from the tensors. Unknown profile identifiers
are preserved, and the raw value remains available in `Metadata`.

Files must contain the tensor data and header padding, including files without
tensors. The reader checks data bounds and sizes without reading the weights.
Unknown encodings retain their numeric identifier with `ByteSize = None`;
their size and any overlap with the next tensor cannot be validated.

Metadata supports all GGUF types, including empty and nested arrays.
Format errors raise `System.IO.InvalidDataException` with a byte position.
File access errors remain ordinary I/O exceptions. Read limits are configurable:

```fsharp
let options =
    { GgufReadOptions.Default with
        MaxArrayElements = 2_000_000
        MaxMetadataValues = 4_000_000 }

let model = Gguf.readFileWithOptions options "model.gguf"
```

Default limits: strings up to 16 MiB, arrays up to one million elements,
100,000 metadata entries and 100,000 tensors, two million metadata values in
total, array nesting depth of 16, and headers up to 256 MiB. Keys remain limited
to 65,535 bytes and tensor names to 64 bytes.

References: [GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
and [GGML encodings](https://github.com/ggml-org/llama.cpp/blob/master/gguf-py/gguf/constants.py).

## Build and tests

```sh
dotnet build Marmotte.slnx
dotnet test Marmotte.slnx
```

Tests construct small binary files in memory; no model or weight download is
required.
