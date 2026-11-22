# Embedding Lookup

[Back to execution tensors and kernels](../kernels.md).

Learn how token IDs select rows of a stored vocabulary matrix. Prerequisites:
arrays, zero-based indexes and [tensor shapes](../kernels.md#reading-the-formulas).
Run `dotnet fsi examples/Embedding.fsx` from the repository root with .NET 10;
its invented vocabulary uses IDs `[1; 0; 1]`, with no model download.

![Embedding lookup: IDs [2, 0, 2] select vocabulary rows 2, 0 and 2 in order, producing three rows of F32 activations](embedding.png)

[Open the embedding illustration](embedding.png).

Think of the vocabulary matrix as a table whose row number is a token ID.
With IDs `[2; 0; 2]`, copy or decode row `2`, then row `0`, then row `2` again.
The example produces rows `[2; 2; 2]`, `[1; 0; 2]`, `[2; 2; 2]`. The token's
**position** chooses the destination row; its **ID** chooses the weight row.
Repeated IDs therefore produce repeated embeddings, and no summation occurs.
The picture uses F32 weights so that every value is directly visible; quantized
weights require decoding the selected rows to recover their F32 values.

`EmbeddingLookup(weights, tokenIds, destination)` selects vocabulary rows:
`destination[i, column] = weights[tokenIds[i], column]`. It applies no scaling,
normalization or special handling for padding tokens. IDs are zero-based signed
integers in `[0, weights.Rows)` and can repeat or arrive in any order. The caller
chooses the matrix; the kernel does not infer architecture or tensor names.

```fsharp
let tokenIds = [| 7; 0; 7 |]
use activations = backend.Allocate(TensorShape.Create [| tokenIds.Length; weights.Columns |])
backend.EmbeddingLookup(weights, ReadOnlyMemory<int>(tokenIds), activations)
```

Destination must have exactly `[tokenIds.Length; weights.Columns]` dimensions,
including `[1; Columns]` for one token. An empty batch requires `[0; Columns]`
and performs no writes. Both tensors must be live and owned by the executing
backend instance. Shape and ownership errors raise `ArgumentException`, null
tensors raise `ArgumentNullException`, and an out-of-range ID raises
`ArgumentOutOfRangeException` identifying its position. Disposal raises
`ObjectDisposedException`. All arguments and every ID are checked before any
output is written, including for an empty batch.

The CPU lookup decodes each selected row with Core's `RowDecoder`, through
`WeightMatrix.DecodeRow`, which supports every current `TensorEncoding`: F32,
F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, IQ4_NL, Q2_K, Q3_K, Q4_K, Q5_K and
Q6_K. It uses safe spans and explicit
little-endian reads, writing only requested rows directly into the destination,
with no temporary decoded matrix, row or block buffers. Stored weights remain unchanged. `WeightMatrix`
validates the payload length and whole blocks per row before upload; the source
still loads the complete encoded matrix into a managed array. Lookup does not
provide memory mapping or lazy file reads. Floating-point NaN, infinities,
subnormals and signed zeros follow F32 conversion and arithmetic.

The block layouts and reconstruction formulas were checked on 2026-10-25 against
[GGML block definitions](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-common.h)
and [GGML reconstruction formulas](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.c):

| Encoding | Stored block | Reconstruction |
| --- | --- | --- |
| Q4_0 | 32 values, 18 bytes | `scale × (q − 8)` with an F16 scale and four-bit `q`; low nibbles precede high nibbles |
| Q4_1 | 32 values, 20 bytes | `q × scale + minimum` with an F16 scale and minimum and four-bit `q` packed as in Q4_0 |
| Q5_0 | 32 values, 22 bytes | `scale × (q − 16)` with five-bit `q`: fifth bits in a little-endian 32-bit word, bit `j` for value `j`, then low bits as in Q4_0 |
| Q5_1 | 32 values, 24 bytes | `q × scale + minimum` with five-bit `q` stored as in Q5_0 |
| Q8_0 | 32 values, 34 bytes | F16 scale times a signed byte |
| IQ4_NL | 32 values, 18 bytes | `scale × level[q]` with four-bit indices into 16 non-uniform levels from −127 to 113 (`kvalues_iq4nl`) |
| Q2_K | 256 values, 84 bytes | Sixteen groups of 16, each with four-bit scale and minimum factors and two-bit `q`; `d × scale × q − dmin × minimum`, with `d` and `dmin` last |
| Q3_K | 256 values, 110 bytes | Sixteen groups of 16 with signed six-bit scales; `q` is two low bits minus 4 when its separate high bit is clear; `d × scale × q` |
| Q4_K | 256 values, 144 bytes | Eight groups of 32, each with six-bit scale and minimum factors; `d × scale × q − dmin × minimum` |
| Q5_K | 256 values, 176 bytes | Q4_K with a fifth bit per value, stored in 32 bytes between the scales and the low bits |
| Q6_K | 256 values, 210 bytes | Sixteen groups of 16; signed group scales and six-bit values centered at 32; `d × scale × (q − 32)` |

The illustrated [Q2_K, Q3_K and BitNet b1.58 guide](../quantization.md) follows
`twoBitQuant` from packed bits to reconstructed values, and separates K-quant
storage costs from BitNet's ternary alphabet and training method.

`RowDecoder` is an original scalar reference implementation, without unsafe
code or a native dependency. Embedding tests specify bytes and expected values
independently of Marmotte's quantizer, exercising packed fields, signed scales, block and row
boundaries, repeated IDs, and validation before writing. GPU lookup remains an
explicit placeholder. Unsupported encodings on a future backend must raise
`NotSupportedException` before submission.

IDs are consumed or copied before the call returns; callers may immediately
reuse the host ID buffer, but must not mutate it during the call. The CPU
operation completes before returning. An asynchronous backend must copy IDs
before returning and follow the same ordering rules as the other kernels.
