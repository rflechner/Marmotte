# Safetensors Format

Safetensors is an open tensor serialization format maintained by Hugging Face.
It is designed to be simple, fast, and safe to parse: unlike pickle-based
formats, loading a Safetensors file does not execute code.

**Marmotte status: index reader implemented.** `Marmotte.Safetensors` reads and
validates the header and tensor descriptors of one `.safetensors` file without
reading the weights, and the CLI's `info` command displays them.
`SafetensorsTensorSource` copies `F32`, `F16` and `BF16` payloads on request for
[format-neutral tensor access](../README.md#format-neutral-tensor-access), where
`RowDecoder` can decode their rows. Sharded models are not implemented; see
[Marmotte implementation](#marmotte-implementation).

The [upstream format definition](https://github.com/huggingface/safetensors#format)
and [reference implementation](https://github.com/huggingface/safetensors/blob/main/safetensors/src/tensor.rs)
are the authoritative references.

## Imported Illustration

The [imported overview](safetensors-format.png) is retained as background
material. It is not an executable specification: the top diagram misplaces the
8-byte length prefix, the pseudocode has misspelled field names, and its sample
header length and hex dump are inconsistent. Its dtype table is also incomplete.
Use the corrected layout and worked example below when implementing the reader.
No editable SVG source was included in the copied documents.

## File Layout

A `.safetensors` file has three consecutive regions:

1. Header length
2. JSON header
3. Tensor data buffer

```text
+----------------+-------------+--------------------+
| header_length  | JSON header | tensor data buffer |
+----------------+-------------+--------------------+
```

## Header Length

The first 8 bytes are an unsigned 64-bit little-endian integer:

```text
N: uint64 little-endian
```

`N` is the number of bytes in the JSON header that immediately follows.

A parser must validate `N` before allocating memory. Marmotte checks it against
the remaining stream length and a configurable limit, independent of the GGUF
reader's options. The default limit, 100,000,000 bytes, matches
`MAX_HEADER_SIZE` in the reference implementation.

## JSON Header

The next `N` bytes are UTF-8 JSON. The format requires the first byte to be `{`;
there is no magic string or separate version field.

The JSON object maps tensor names to tensor descriptors. The special
`__metadata__` key is optional and contains string-to-string metadata.

Example:

```json
{
  "tensor": {
    "dtype": "F16",
    "shape": [2, 3],
    "data_offsets": [0, 12]
  },
  "__metadata__": {
    "format": "pt"
  }
}
```

The JSON header may end with ASCII spaces, included in `N`; the reference writer
pads the header to a multiple of 8 bytes. The reference reader leaves this
padding to its JSON parser, which accepts any JSON whitespace. Marmotte does the
same: spaces, tabs, carriage returns and line feeds may follow the object, but
no other byte. Tensor names are object keys. `__metadata__` is a string-to-string map, not a tensor descriptor.
There is no extra GGUF-style alignment region between the header and data.

## Tensor Descriptor

Each tensor descriptor contains:

```text
dtype:        string
shape:        JSON array of nonnegative integers
data_offsets: [BEGIN, END]
```

`BEGIN` is inclusive and `END` is exclusive. Offsets are relative to the start of
the tensor data buffer, not to the start of the file:

```text
data_start = 8 + N
absolute_position = data_start + BEGIN
byte_count = END - BEGIN
```

Marmotte exposes dimensions and offsets as `uint64`, an F# API choice rather
than a binary encoding inside the JSON. It rejects negative, fractional,
exponent-form, out-of-range, or incorrectly shaped JSON values before conversion.
When parsing from a nonzero stream position, it adds that starting position to
`data_start` in the absolute stream offsets it exposes.

### Shapes and Element Sizes

Shape `[]` describes a scalar with one element. A zero dimension describes an
empty tensor with no payload. These are valid cases, not malformed shapes.

For supported byte-sized dtypes, the expected byte count is the checked product
of dimensions multiplied by bytes per element. Do not confuse `F16` with
`BF16`: both occupy two bytes but have different representations.

The current upstream implementation also includes FP8 and sub-byte dtypes.
Its size check uses bits, requiring a whole number of bytes:

```text
element_count = product(shape)     // product([]) = 1
bit_count = element_count * bits_per_element
require bit_count % 8 == 0
require END - BEGIN == bit_count / 8
```

Marmotte supports the 22 dtypes of the reference implementation's `Dtype` enum,
with the widths of `Dtype::bitsize`
([tensor.rs](https://github.com/huggingface/safetensors/blob/main/safetensors/src/tensor.rs),
checked on 2026-10-24):

| Bits per element | Dtypes |
| ---: | --- |
| 4 | `F4` |
| 6 | `F6_E2M3`, `F6_E3M2` |
| 8 | `BOOL`, `U8`, `I8`, `F8_E5M2`, `F8_E4M3`, `F8_E8M0`, `F8_E4M3FNUZ`, `F8_E5M2FNUZ` |
| 16 | `I16`, `U16`, `F16`, `BF16` |
| 32 | `I32`, `U32`, `F32` |
| 64 | `C64` (two 32-bit floating-point parts), `F64`, `I64`, `U64` |

Names are case-sensitive. Other dtypes are rejected: a packed size is never
rounded and an unknown dtype is never reinterpreted. Update this table and
`SafetensorsDtype` together when upstream adds a dtype.

## Tensor Data Buffer

The final region contains raw packed tensor bytes.

Important properties:

- Numeric values are little-endian.
- Tensor data is stored in row-major / C order.
- The format stores no strides.
- Tensor data can be exposed lazily as slices of the original file.

The last dimension is contiguous. A matrix with two rows and three columns
has shape `[2; 3]` and element strides `[3; 1]`, unlike the GGML dimension order
preserved by the GGUF reader. See the [storage and stride notes](README.md).

Lazy access requires an explicit owner for the file or mapping. Avoid promising
zero-copy GPU transfers or allocation-free header parsing. Such behavior belongs
to the loading implementation, not merely the file format.

## Worked Example

This compact JSON header is exactly **112 UTF-8 bytes**, with no newline or
trailing spaces:

```json
{"a":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]},"b":{"dtype":"I32","shape":[3],"data_offsets":[24,36]}}
```

The length prefix contains 112 (`70 00 00 00 00 00 00 00` in hexadecimal).
The complete file is 156 bytes:

| File byte range | Contents |
| --- | --- |
| `[0, 8)` | Header length, little-endian `uint64` |
| `[8, 120)` | JSON header, 112 bytes |
| `[120, 144)` | `a`: six F32 values, 24 bytes |
| `[144, 156)` | `b`: three I32 values, 12 bytes |

No additional bytes are inserted between these regions. Adding spaces to the
JSON changes `N` and both absolute tensor positions, but not `data_offsets`.

## Safe Parsing Order

1. Read exactly 8 bytes.
2. Decode `N` as `uint64` little-endian.
3. Validate `N` against configured limits and file length.
4. Read `N` bytes.
5. Validate the initial `{`, strict UTF-8, JSON object structure, and trailing padding.
6. Reject duplicate decoded JSON keys, including inside descriptors and metadata.
7. Validate every tensor descriptor.
8. Compute `data_start = 8 + N`.
9. Validate `data_offsets`.
10. Sort ranges by `(BEGIN, END)` and validate their sizes and contiguity.
11. Require exact coverage of the remaining file, without holes or trailing data.
12. Return a validated index; add lazy slices later with explicit ownership.

Do not rely on JSON property order. Start the coverage cursor at zero; every
sorted range must start at the cursor and move it to `END`. Zero-length ranges
can share a boundary, and an empty data buffer is valid when there are no
nonempty tensors.

## Security Requirements

Safetensors files must be treated as untrusted input.

- Use checked arithmetic for all offset and size calculations.
- Validate header size, tensor counts, rank, and JSON depth before allocation.
- Validate UTF-8 and JSON structure.
- Reject duplicate keys.
- Validate `dtype`, `shape`, and `data_offsets`.
- Reject negative, reversed, overlapping, or out-of-file ranges.
- Confirm the encoded byte size agrees with the shape and dtype layout.
- Never trust offsets merely because the JSON parses successfully.

## Marmotte Implementation

`Marmotte.Safetensors` mirrors the GGUF reader's index-inspection scope, without
a tensor runtime. Its dtypes use a dedicated `SafetensorsDtype` type, distinct
from `GgmlType` and `GgufFileType`.

```fsharp
open Marmotte.Safetensors

let model = Safetensors.readFile "model.safetensors"
printfn "%d tensors, data buffer at byte %d" model.Tensors.Length model.DataOffset
```

| Property | Current contract |
| --- | --- |
| Input | A readable, seekable stream, from its current position to its end |
| Stream ownership | `read` leaves the caller's stream open at `DataOffset`; `readFile` opens and closes its own |
| Weights | Never read; ranges are validated against the stream length |
| Tensor order | Data offset, then end offset and name; JSON order is not preserved |
| Shapes | Row-major as stored, contiguous dimension last |
| Metadata | `__metadata__` as a string map, empty when absent |
| Duplicate keys | Rejected after JSON unescaping, for tensors, descriptor fields and metadata |
| Descriptor fields | `dtype`, `shape` and `data_offsets` are required; other fields are rejected |
| JSON | Strict UTF-8, leading `{`, no comments or trailing commas, nesting depth of at most 3 |
| Limits | `SafetensorsReadOptions`: header bytes (100,000,000), tensors (100,000), rank (64) |
| Errors | `InvalidDataException` for malformed files; I/O exceptions propagate |

Some checks are stricter than the reference reader. It does not check the leading
`{` that the format requires, and leaves duplicate keys to its JSON library;
Marmotte checks both explicitly. Its deserialization ignores unknown descriptor
fields and accepts a `null` `__metadata__`; Marmotte rejects them, so that a
future field cannot silently change the meaning of a payload.

The CLI's `info` command recognizes a Safetensors file from its content and
reports the header and data buffer sizes, a summary by dtype, metadata, and a
bounded tensor preview. Tests encode headers and data buffers directly and cover
the worked example, scalars, empty tensors, reordered JSON entries, padding,
every dtype width, duplicate keys, truncation, invalid sizes and coverage errors.

`SafetensorsTensorSource.openFile` keeps the file open until disposal and exposes
the index through `Marmotte.Core`'s `ITensorSource`. Shapes are copied unchanged,
since both use row-major order. Only `F32`, `F16` and `BF16` map to a runtime
encoding; other dtypes stay listed with `Encoding = None`, and reading their
payload raises `InvalidDataException`. A shard of a sharded model exposes only
its own tensors.

Not implemented yet:

- Sharded models described by `model.safetensors.index.json`, and companion files
  such as `config.json` and tokenizer files.
- Memory-mapped payloads, shared tensor storage and inference, which belong to
  later milestones.
- An exploration `.fsx` script; the `info` command covers inspection.

## References

- [Hugging Face Safetensors documentation](https://huggingface.co/docs/safetensors/index)
- [Safetensors GitHub repository](https://github.com/huggingface/safetensors)
- [Format definition](https://github.com/huggingface/safetensors#format)
- [Dtypes and range validation](https://github.com/huggingface/safetensors/blob/main/safetensors/src/tensor.rs)
