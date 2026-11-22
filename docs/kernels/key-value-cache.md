# Key-Value Cache

[Back to execution tensors and kernels](../kernels.md).

Learn how cached keys and values avoid recomputing an earlier token prefix.
Prerequisites: [tensor shapes](../kernels.md#reading-the-formulas). This page
introduces the stored keys and values; continue with the
[attention walkthrough](attention.md) to see how queries use them.
Run `dotnet fsi examples/KeyValueCache.fsx` from the repository root with .NET 10
to print cache sizes and round-trip errors without a model file.

Attention is the only operation in which a token reads the other positions: its
query is compared with the keys of every earlier position, and their values are
weighted by the result. Everything else in a block is computed token by token.
Without a cache, generating a token would recompute the key and value
projections of the whole prefix in every layer. The cache stores them instead,
once per position and layer. Queries are not stored: a past token asks nothing
more.

`KeyValueCacheLayout` describes the cache: its layers, its capacity in
positions, its key-value heads and their dimension, and one encoding for keys
and another for values. Each position of a layer is a row of
`KeyValueHeads × HeadDimension` values for keys and another for values, 1,024
values for the target model:

| Encoding | Bytes per 32 values | Per token | 4,096 tokens | 32,768 tokens |
| --- | --- | --- | --- | --- |
| F32 | 128 | 208 KiB | 832 MiB | 6,656 MiB |
| F16, BF16 | 64 | 104 KiB | 416 MiB | 3,328 MiB |
| Q8_0 | 34 | 55.25 KiB | 221 MiB | 1,768 MiB |
| Q5_1 | 24 | 39 KiB | 156 MiB | 1,248 MiB |
| Q5_0 | 22 | 35.75 KiB | 143 MiB | 1,144 MiB |
| Q4_1 | 20 | 32.5 KiB | 130 MiB | 1,040 MiB |
| IQ4_NL, Q4_0 | 18 | 29.25 KiB | 117 MiB | 936 MiB |

```fsharp
let layout =
    KeyValueCacheLayout.Create(
        layers = 26,
        capacity = 4096,
        keyValueHeads = 8,
        headDimension = 128,
        keyEncoding = TensorEncoding.Q8_0,
        valueEncoding = TensorEncoding.Q8_0)

use cache = backend.AllocateKeyValueCache(layout)

// In each layer: rotate the keys, then store keys and values at their positions.
backend.Rope(keys, positions, embedding, keys)
backend.StoreKeyValues(cache, layer, firstPosition, keys, values)
```

`BytesPerPosition` and `TotalBytes` give these sizes before allocating anything.
Keys and values have separate encodings, like llama.cpp's `--cache-type-k` and
`--cache-type-v` options, which LM Studio exposes as K and V cache quantization.
Marmotte supports the same encodings as llama.cpp: F32, F16, BF16, Q8_0, Q5_1,
Q5_0, Q4_1, IQ4_NL and Q4_0. llama.cpp's default is F16. K-quants such as Q4_K
are rejected: their reference quantizers search for sub-block scales, which
Marmotte does not implement. A head dimension must hold whole blocks, so a
32-value block never spans two heads and attention can decode one head at a
time.

Unlike llama.cpp, a quantized value cache does not require flash attention.
Without flash attention, llama.cpp stores values transposed, with positions
contiguous, so that blocks would have to span positions; Marmotte stores every
position as a row of whole blocks, for values as for keys.

`RowEncoder` in Core writes the rows; it is the inverse of `RowDecoder` and also
encodes the GGUF quantizer's blocks. F16 and BF16 round to nearest, ties to
even. Q8_0 rounds each value to half a step of `max / 127`, where `max` is the
largest magnitude of its block of 32. Q5_0 and Q4_0 use 32 and 16 steps of
`max / 16` and `max / 8`, and can clip the end opposite to their scale by a
whole step. Q5_1 and Q4_1 also store the block's minimum and divide its range
into 31 and 15 steps, which suits blocks whose values are not centered on zero.
IQ4_NL has the size of Q4_0 but 16 levels closer together near zero, where most
values lie: its encoder tries 16 scales and keeps the one with the smallest
error weighted by `x²`, as llama.cpp's `quantize_iq4_nl` does without an
importance matrix, then picks each index against the F16 scale actually stored
rather than the F32 one. On normally distributed values, its squared error is
about 20% below Q4_0's. Every encoding stores its scale and minimum in F16.
Quantization is lossy, and the precision attention needs is a measurement
to make on the model, not a property of the cache: keys have outlier channels,
which inflate the scale of every block that contains one
([KIVI](https://arxiv.org/abs/2402.02750)), so they usually tolerate
quantization less well than values. Q8_0 for both is a common starting point,
and a lower-precision value cache the next step.

`StoreKeyValues(cache, layer, position, keys, values)` encodes F32 tensors of
shape `[tokens; KeyValueHeads × HeadDimension]` or
`[tokens; KeyValueHeads; HeadDimension]` into positions `position` to
`position + tokens - 1`. Each layer stores consecutive positions from 0, counted
by `StoredPositions(layer)`: `position` may not leave a gap. Storing below that
count replaces those positions and discards every later one, which was
computed from the replaced tokens. Regenerating a response from position `p`
therefore needs no separate truncation, and an empty batch at `p` only
truncates. `position + tokens` may not exceed the capacity.

`ReadKeyValues(cache, layer, position, keys, values)` decodes stored positions
into two different F32 tensors of the same shapes. An F32 cache reads back bit
for bit; another encoding reads back what `RowDecoder` decodes from the stored
blocks. Attention decodes cached heads directly, without materializing all
stored rows as execution tensors.

Values that an encoding cannot hold are never replaced by plausible numbers.
F16 and BF16 turn values beyond their range into infinities, and NaN stays NaN.
A quantized block that contains NaN or an infinity, or whose scale or minimum
exceeds the F16 range, is stored with a NaN scale: its 32 values read back as NaN, and
the other blocks are unaffected. The GGUF quantizer is stricter and rejects
such weights.

The CPU backend allocates the whole capacity up front, as llama.cpp does: one
byte array of `Capacity` rows per layer for keys and one for values. Blocks
never span rows, so a batch of positions is encoded or decoded in one call. A
layer's keys or values must fit in a managed array: 524,287 F32 positions for
the target model, twice its 262,144-position context. Tests check
hand-computed bytes of every encoding, rounding errors, non-finite blocks, the target model's sizes,
appending, replacing and truncating positions, validation before any write,
ownership and disposal. Run
[`examples/KeyValueCache.fsx`](../../examples/KeyValueCache.fsx) to print the sizes
above and the error of each encoding.
