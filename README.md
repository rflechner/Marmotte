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

## Safetensors reader

`Marmotte.Safetensors` reads the index of a single `.safetensors` file: the JSON
header, its string metadata and the tensor descriptors, without reading the
weights.

```fsharp
open Marmotte.Safetensors

let model = Safetensors.readFile "model.safetensors"

for tensor in model.Tensors do
    printfn "%s: %s %A, position %d"
        tensor.Name tensor.Dtype.WireName tensor.Shape tensor.AbsoluteOffset
```

`Safetensors.read stream` accepts a readable, seekable stream, starting at its
current position, and leaves it open at `DataOffset`. Shapes are in row-major (C)
order, with the contiguous dimension last, unlike GGUF dimensions. `Offset` is
relative to the tensor data buffer; `AbsoluteOffset` and `DataOffset` are
positions in the stream. Tensors are returned in data offset order, not in JSON
order. `Metadata` contains the optional `__metadata__` strings.

The reader validates the header length, UTF-8, the JSON structure, the 22 dtypes
of the reference implementation, shapes, sizes and offsets. The tensors must cover
the data buffer exactly, without gaps, overlaps or trailing bytes. Duplicate keys,
compared after JSON unescaping, unknown dtypes and unknown descriptor fields are
rejected rather than ignored. Format errors raise `System.IO.InvalidDataException`;
file access errors remain ordinary I/O exceptions. `SafetensorsReadOptions`
configures the limits: by default a 100,000,000-byte header, as in the reference
implementation, 100,000 tensors and rank 64.

Sharded models described by `model.safetensors.index.json` are not handled yet;
inspect each shard separately. See the
[Safetensors format and implementation notes](docs/safetensors.md).

## Format-neutral tensor access

`Marmotte.Core` describes tensors independently of their file format.
`Marmotte.Gguf` and `Marmotte.Safetensors` reference it and implement the
mapping, so that code running a model reads both formats through one contract:

```fsharp
open Marmotte.Core
open Marmotte.Gguf

use source = GgufTensorSource.openFile "model.gguf"
let query = WeightMatrix.Read(source, "blk.0.attn_q.weight")

printfn "%A: %d rows of %d values, %d bytes per row"
    query.Encoding query.Rows query.Columns query.RowBytes
```

`SafetensorsTensorSource.openFile` returns the same `ITensorSource`. Descriptors
use row-major shapes, with the contiguous dimension last: the GGUF mapping
reverses its dimensions, which describe the same bytes. A `WeightMatrix` has
`Rows` and `Columns`, where `Columns` is the contiguous dimension, and exposes
each encoded row as a `ReadOnlySpan<byte>`. Quantization blocks never cross rows.

`TensorEncoding` lists the encodings Marmotte represents at runtime: `F32`,
`F16`, `BF16`, `Q4_0`, `Q8_0`, `Q4_K` and `Q6_K`. Tensors in other encodings
remain listed, with `Encoding = None` and their source name in `SourceEncoding`;
reading their payload raises `InvalidDataException`. Opening a big-endian or
split GGUF file raises it too.

The source keeps its file open until it is disposed. Each payload request copies
the tensor's validated byte range into a managed array with positional reads,
which can run concurrently. A payload must therefore fit in one array, and
loaded weights occupy as much memory as in the file. Memory mapping, row decoding,
architecture-specific tensor names and model configuration are not implemented yet.

## Inspect a local GGUF file

### Command line

Run `info` with an absolute file path or a path relative to the current working
directory:

```sh
dotnet run --project src/Marmotte.Cli -- info "/path/to/model.gguf"
dotnet run --project src/Marmotte.Cli -- info "/path/to/model.gguf" --metadata --tensor-limit 20
dotnet run --project src/Marmotte.Cli -- info "/path/to/model.safetensors" --metadata
dotnet run --project src/Marmotte.Cli -- info --help
```

`info` detects the format from the file's content, not its extension: a GGUF
file starts with its magic, a Safetensors file with an 8-byte length followed by
the `{` of its JSON header.

The command displays the model name, architecture, file-level quantization
profile, GGUF version and byte order, file size, metadata and tensor counts,
total tensor elements, alignment, tensor data offset, and index parsing time.
It also groups tensors by encoding and previews up to 12 tensor descriptors.
Dimensions remain in GGML order, with the contiguous dimension first; tensor
offsets in the report are absolute file positions. Unknown encodings retain their
numeric identifier and display an unknown byte size.

Use `--metadata` to show all metadata entries. String previews are limited to
120 characters, array previews to three elements, and nested array previews to
two levels. `--tensor-limit COUNT` changes the number of tensor descriptors
displayed; `0` hides the preview. The parser reads only metadata and tensor
descriptors, without loading or dequantizing weights.

For a Safetensors file, the report shows the file and JSON header sizes, metadata
and tensor counts, total tensor elements, the position and size of the tensor
data buffer, and index parsing time. It groups tensors by dtype and previews up
to 12 tensors in data offset order. Shapes keep the row-major order of the file,
with the contiguous dimension last. `--metadata` shows the `__metadata__` strings
with the same 120-character previews.

Successful inspection exits with code `0`. Invalid arguments, inaccessible files,
and malformed or unrecognized files produce a diagnostic on stderr and exit with
code `1`.

### Launch profile for Ministral

[launchSettings.json](src/Marmotte.Cli/Properties/launchSettings.json) includes a
`Ministral-3B` profile that runs `info --metadata` on the Ministral 3 3B Instruct
2512 Q4_K_M model also selected by the exploration script. Set
`LM_STUDIO_MODELS_FOLDER` to your local models directory before launching:

```sh
dotnet run --project src/Marmotte.Cli --launch-profile Ministral-3B
```

The profile inherits the variable from the launching process and expands
`%LM_STUDIO_MODELS_FOLDER%` in its quoted file argument, so paths containing
spaces work. The personal root directory stays outside the repository.
This is the default launch profile; explicit application arguments after `--`
take precedence over its arguments.

In Rider, select the `Ministral-3B` launch profile to run or debug it. If it is
missing from existing run configurations, right-click `launchSettings.json` and
choose **Generate Configurations**, as described in the
[Rider launch profile documentation](https://www.jetbrains.com/help/rider/Running_LaunchSettings.html).
The IDE must have `LM_STUDIO_MODELS_FOLDER` in its environment when it launches
the application.

### F# exploration script

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

## Quantize a GGUF file

The native F# `quantize` command converts a floating-point GGUF into a **new**
GGUF file. It does not require llama.cpp or a native executable:

```sh
dotnet run --project src/Marmotte.Cli -- quantize model-f16.gguf model-q8_0.gguf
dotnet run --project src/Marmotte.Cli -- quantize model-f16.gguf model-q4_0.gguf --type Q4_0
dotnet run --project src/Marmotte.Cli -- quantize --help
```

`--type` accepts `Q8_0` (default) or `Q4_0`, case-insensitively. These are lossy
conversions; Q4_0 uses fewer bytes per block and has lower precision than Q8_0.
The command reports converted/preserved tensor counts, input/output sizes and
elapsed time. Success exits with code `0`; errors exit with code `1`.
Ctrl+C requests cancellation, cleans up the temporary output and exits with code `130`.
A second Ctrl+C terminates the process immediately and can leave the temporary
`.marmotte-*.tmp` file in the output directory.

This first implementation supports:

- A complete, single-file, **little-endian GGUF v2 or v3**, containing only
  `F32`, `F16` and/or `BF16` tensors.
- Conversion of non-empty tensors of rank 2 or greater whose first dimension
  (the contiguous GGML dimension) is divisible by 32. Scalars, vectors, empty
  tensors and other row widths retain their original bytes and encoding.
- GGUF v3 little-endian output, preserving names, shapes, alignment and typed
  metadata, while updating `general.file_type` and `general.quantization_version`.
- Streaming weights with less than 128 KiB of reusable payload buffers, in
  addition to the bounded metadata/index allocations.

The input is opened read-only. Existing output files are never overwritten.
A temporary file in the output directory is validated and moved to the requested
path only after conversion succeeds; failures and handled cancellation remove
the temporary file. The output directory must already exist. Files with no
eligible tensors are rejected.

Requantization, split files, big-endian input, and output profiles such as
`Q3_K_S` or `Q4_K_M` are **not implemented**. The quantizer checks actual tensor
encodings rather than trusting the filename or file-level profile. The Ministral
`Q4_K_M` used by the inspection launch profile is therefore not a supported input;
select a floating-point GGUF instead. Files under LM Studio can be addressed with
`"$LM_STUDIO_MODELS_FOLDER/<relative-model-path>.gguf"`.

The library API is also available independently of the CLI:

```fsharp
let result =
    GgufQuantizer.quantizeFile GgufQuantizationFormat.Q8_0
        "model-f16.gguf" "model-q8_0.gguf"
```

`quantizeFileWithCancellation` accepts the same arguments followed by a
`System.Threading.CancellationToken`. Unsupported or malformed data, NaN/infinite
converted weights, and scales outside the finite F16 range raise
`InvalidDataException`; file access errors remain ordinary I/O exceptions.
See [native quantization details](docs/gguf.md#native-quantization) for the
block layouts, numerical limits and tensor selection policy.

## Documentation

- [Format comparison, tensor storage, strides, and illustration notes](docs/README.md)
- [Illustrated inference pipeline: model files, embeddings, KV cache and generation](docs/inference-pipeline.md)
- [GGUF format and current implementation](docs/gguf.md)
- [Safetensors format and current implementation](docs/safetensors.md)
- [Roadmap](ROADMAP.md): CPU inference first, then CUDA and Metal backends

## Build and tests

NuGet package versions are managed centrally in
[Directory.Packages.props](Directory.Packages.props). Project files declare
`PackageReference` entries without versions, including an explicit `FSharp.Core`
reference in each F# project because CPM disables the SDK's implicit reference.
`Marmotte.Cli` references
`Spectre.Console` for console rendering and `Spectre.Console.Cli` for command-line
parsing.

```sh
dotnet build Marmotte.slnx
dotnet test Marmotte.slnx
```

Parser and quantizer tests construct small binary fixtures and check index
validation, exact quantization blocks, numerical error, metadata preservation,
and output protection. Safetensors tests encode JSON headers and data buffers
directly. CLI tests run the application against temporary GGUF and Safetensors
fixtures and check both commands, argument handling, and errors. Core tests
check tensor sources and weight matrices with an in-memory source and temporary
files.
No model or weight download is required.

## Coding guidelines

Coding conventions are described in [AGENTS.md](AGENTS.md). Marmotte uses safe
.NET code by default: arrays, spans and `BinaryPrimitives`. Unsafe code, such as
native pointers, `fixed` or `System.Runtime.CompilerServices.Unsafe`, is reserved
for cases where a measurement shows a large performance gain. It then stays
confined behind a safe API, with bounds validated beforehand, and is tested
against a safe reference implementation.
