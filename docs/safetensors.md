# Safetensors Format

Safetensors is an open tensor serialization format maintained by Hugging Face.
It is designed to be simple, fast, and safe to parse: unlike pickle-based
formats, loading a Safetensors file does not execute code.

**Marmotte status: planned, not implemented.** This is a format reference and a
proposed implementation plan. There is no `Marmotte.Safetensors` project yet.
The existing GGUF reader does not read Safetensors files.

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

A parser must validate `N` before allocating memory. Marmotte's future reader
should enforce configurable header and nesting limits, independently of the
GGUF reader's options.

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

The JSON header may end with ASCII spaces, included in `N`. Tensor names are
object keys. `__metadata__` is a string-to-string map, not a tensor descriptor.
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

The `uint64` representation for dimensions and offsets is a proposed F# API
choice, not a binary integer encoding inside the JSON. Reject negative,
fractional, out-of-range, or incorrectly shaped JSON values before conversion.
If parsing from a nonzero stream position, add that starting position to
`data_start` when exposing absolute stream offsets.

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

Verify the dtype table against the targeted upstream version before coding.
For the initial reader, explicitly list supported dtypes and reject others;
do not silently round a packed size or reinterpret an unknown dtype.

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

## Proposed Marmotte Implementation Plan

The first milestone should mirror GGUF's existing index-inspection scope,
without requiring a tensor runtime or GPU backend:

1. Add `Marmotte.Safetensors` and a dedicated test project to `Marmotte.slnx`.
2. Model file metadata and tensor descriptors with F# records, and dtypes with a
   dedicated type. Keep wire tags distinct from `GgmlType` and `GgufFileType`.
3. Implement bounded stream/file readers with documented position, ownership,
   and error contracts consistent with the GGUF API where appropriate.
4. Use `System.Text.Json`; detect duplicate decoded names before building maps
   instead of relying on last-value-wins property lookup.
5. Validate dtype widths, shapes, ranges, overflow, and complete data coverage.
   Return metadata and descriptors without reading the weights.
6. Add synthetic fixtures for ordinary tensors, scalars, empty tensors,
   reordered JSON entries, padding, duplicate keys, truncation, and invalid sizes.
7. Add a small `.fsx` inspection script, then optionally extend `Marmotte.Cli`.
   Do not create separate GGUF and Safetensors CLI projects without a need.

Bounded payload access, mapping lifetimes, shared tensor storage, decoding,
and inference belong to later milestones. Multi-file shard indexes are also a
separate concern from parsing one `.safetensors` file.

## References

- [Hugging Face Safetensors documentation](https://huggingface.co/docs/safetensors/index)
- [Safetensors GitHub repository](https://github.com/huggingface/safetensors)
- [Format definition](https://github.com/huggingface/safetensors#format)
- [Dtypes and range validation](https://github.com/huggingface/safetensors/blob/main/safetensors/src/tensor.rs)
