# GGUF Format

GGUF is a binary container format used to store model metadata, tensor
descriptors, and tensor data in a single file. DotDiffusion uses it as one of
its model-loading formats, especially for quantized weights.

This document focuses on the information required to implement a safe,
incremental GGUF parser.

![GGUF file format parser map](gguf-format.png)

## File Layout

A GGUF file contains five consecutive regions:

1. Header
2. Metadata key-value pairs
3. Tensor descriptors
4. Alignment padding
5. Tensor data

Except for the padding before tensor data, fields are serialized consecutively
without implicit alignment.

```text
+---------+------------------+--------------+---------+-------------+
| Header  | Metadata entries | Tensor infos | Padding | Tensor data |
+---------+------------------+--------------+---------+-------------+
```

Integers use the byte order detected from the version field. GGUF files are
normally little-endian, while GGUF v3 can also represent big-endian data.

## Header

The header occupies exactly 24 bytes.

| Offset | Size | Type | Field |
| ---: | ---: | --- | --- |
| `0` | 4 | ASCII bytes | Magic: `GGUF` |
| `4` | 4 | `uint32` | Format version |
| `8` | 8 | `uint64` | Number of tensors |
| `16` | 8 | `uint64` | Number of metadata entries |

DotDiffusion currently supports GGUF versions 2 and 3.

The version is read in both byte orders. A supported value identifies the byte
order to use for the remaining numeric fields. Invalid magic bytes, unsupported
versions, and truncated headers are rejected.

The parsed representation is:

```csharp
public readonly record struct GgufHeader(
    uint Version,
    ulong TensorCount,
    ulong MetadataKeyValueCount,
    GgufByteOrder ByteOrder);
```

The current implementation lives in
[`GgufHeaderReader`](../src/DotDiffusion.Gguf/Readers/GgufHeaderReader.cs).

## Strings

GGUF strings are length-prefixed UTF-8 values:

```text
length: uint64
data:   byte[length]
```

There is no null terminator.

A parser must validate the length before allocating memory, ensure the complete
byte sequence is available, and reject invalid UTF-8 if strict decoding is
used.

## Metadata

The header's metadata count determines how many key-value entries follow.

Each entry contains:

```text
key:        STRING
value_type: uint32
value:      encoded according to value_type
```

### Value Types

| Value | Name | Encoded value |
| ---: | --- | --- |
| `0` | `UINT8` | `uint8` |
| `1` | `INT8` | `int8` |
| `2` | `UINT16` | `uint16` |
| `3` | `INT16` | `int16` |
| `4` | `UINT32` | `uint32` |
| `5` | `INT32` | `int32` |
| `6` | `FLOAT32` | IEEE 754 binary32 |
| `7` | `BOOL` | `int8`, zero or one |
| `8` | `STRING` | GGUF string |
| `9` | `ARRAY` | Element type, count, then elements |
| `10` | `UINT64` | `uint64` |
| `11` | `INT64` | `int64` |
| `12` | `FLOAT64` | IEEE 754 binary64 |

An array is encoded as:

```text
element_type: uint32
count:        uint64
elements:     value[count]
```

Arrays cannot contain arrays. The element type determines the encoding of every
element.

Common metadata keys include model architecture, model name, tokenizer data,
quantization information, and `general.alignment`.

DotDiffusion represents decoded metadata values with an abstract record
hierarchy rather than a public `object` payload plus type tag. This keeps
callers on the type-safe path:

```csharp
GgufMetadataValue value = entry.Value;

string? architecture = value switch
{
    GgufStringMetadataValue stringValue => stringValue.Value,
    _ => null,
};
```

Each GGUF scalar type has a dedicated sealed record, such as
`GgufUInt32MetadataValue`, `GgufStringMetadataValue`, and
`GgufFloat64MetadataValue`. Arrays are represented by
`GgufArrayMetadataValue`, whose `Values` collection contains already decoded
`GgufMetadataValue` instances. GGUF does not allow arrays of arrays.

## Tensor Descriptors

The tensor count from the header determines how many tensor descriptors follow
the metadata entries.

Each descriptor contains:

```text
name:          STRING
n_dimensions:  uint32
dimensions:    uint64[n_dimensions]
ggml_type:     uint32
offset:        uint64
```

The offset is relative to the beginning of the tensor data region, not to the
beginning of the file:

```text
absolute_position = tensor_data_start + tensor.offset
```

The tensor's byte size cannot be computed from its dimensions alone. It also
depends on the `ggml_type`, including its block size and encoded bytes per
block for quantized types.

DotDiffusion currently parses tensor descriptors into `GgufTensorInfo` values:

```csharp
public sealed record GgufTensorInfo(
    string Name,
    IReadOnlyList<ulong> Dimensions,
    GgmlType Type,
    ulong Offset);
```

The offset remains relative to the tensor data region. Resolving it to an
absolute file position happens later, after alignment is known.

## Alignment

The tensor data region starts at an aligned file position.

The infographic labels this region as "Alignment padding". This is a
DotDiffusion documentation label, not an official GGUF field name. In the GGUF
specification this is represented as `_padding[]`: zero bytes inserted after the
tensor infos so that `tensor_data` begins at the required alignment.

The default alignment is 32 bytes. A `general.alignment` metadata entry can
override it.

```text
tensor_data_start = align(position_after_tensor_infos, alignment)
```

For a power-of-two alignment:

```text
align(x, a) = (x + a - 1) & ~(a - 1)
```

The parser must validate that the alignment is non-zero and suitable before
using this formula. All arithmetic must be checked for overflow.

Tensor offsets must respect the selected alignment.

## Tensor Data

The final region contains raw or quantized tensor bytes. A parser should expose
this data lazily rather than copying the complete model into managed memory.

The intended loading strategy is:

- Keep the file open or memory-map it.
- Parse metadata and tensor descriptors eagerly.
- Validate every tensor range before exposing it.
- Represent tensor payloads as bounded slices of the underlying storage.
- Decode or dequantize values only when an execution backend needs them.

## Safe Parsing Order

1. Read exactly 24 header bytes.
2. Validate the `GGUF` magic.
3. Detect byte order and validate the version.
4. Read tensor and metadata counts.
5. Validate counts against configured limits.
6. Parse each typed metadata entry.
7. Read `general.alignment`, or use 32.
8. Parse each tensor descriptor.
9. Compute the aligned tensor data start.
10. Validate tensor offsets and encoded sizes against the file length.
11. Expose immutable metadata and lazy tensor slices.

## Security Requirements

GGUF files must be treated as untrusted input.

- Use checked arithmetic for additions, multiplications, and alignment.
- Set explicit limits for strings, arrays, dimensions, metadata, and tensors.
- Validate a length before allocating or slicing.
- Reject unknown metadata value types.
- Reject invalid dimensions and unsupported tensor types.
- Verify that every tensor range remains inside the file.
- Detect overlapping tensor ranges unless the format explicitly permits them.
- Never convert an untrusted `ulong` directly to `int` or `nint`.
- Do not trust offsets merely because the file has a valid header.

## DotDiffusion Implementation Status

Implemented:

- Fixed-size header reading
- Magic validation
- GGUF v2 and v3 validation
- Little-endian and big-endian detection
- Tensor and metadata counts
- Truncated-header detection
- Endian-aware primitive reading
- Bounded UTF-8 string reading
- Metadata scalar values
- Type-safe metadata value records
- Metadata arrays
- Tensor descriptors
- Unit tests for valid and invalid headers

Planned next:

1. Resolve alignment and tensor data positions.
2. Add lazy, validated tensor data access.

## References

- [GGUF format documentation](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
- [GGUF public definitions](https://github.com/ggml-org/ggml/blob/master/include/gguf.h)
- [Hugging Face Hub GGUF documentation](https://huggingface.co/docs/hub/gguf)
