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

## Inspect a local GGUF file

Set `LM_STUDIO_MODELS_FOLDER` to your local LM Studio models directory, then run
the F# script from the repository root. For example:

```sh
export LM_STUDIO_MODELS_FOLDER="$HOME/.lmstudio/hub/models"
dotnet fsi src/Marmotte.Gguf/GgufScript.fsx
```

Use the actual models directory configured on your machine. This variable holds
the root directory; the script's `filePath` binding appends these relative path
segments using the `/>` helper (`Path.Combine`):

```text
lmstudio-community/Ministral-3-3B-Instruct-2512-GGUF/Ministral-3-3B-Instruct-2512-Q4_K_M.gguf
```

To inspect another model, change those relative segments in the script. Its
opening message prints only the filename. The script inspects metadata and tensor
descriptors without loading the weights.

You can keep the `export` in your local `~/.zprofile`. After editing it, run
`source ~/.zprofile` in the terminal before starting `dotnet fsi`. For an IDE,
make sure its environment includes the variable before it starts F# Interactive;
an already-running process does not pick up shell environment changes.

Alternatively, set the variable directly in an F# Interactive session with
`System.Environment.SetEnvironmentVariable("LM_STUDIO_MODELS_FOLDER", "/path/to/models")`
before loading the script. Keep this personal setting outside tracked files.

## Documentation

- [Format comparison, tensor storage, strides, and illustration notes](docs/README.md)
- [GGUF format and current implementation](docs/gguf.md)
- [Safetensors format and proposed next steps](docs/safetensors.md) — not implemented yet

## Build and tests

```sh
dotnet build Marmotte.slnx
dotnet test Marmotte.slnx
```

Tests construct small binary files in memory; no model or weight download is
required.
