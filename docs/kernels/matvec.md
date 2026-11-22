# Matrix-Vector Product

[Back to execution tensors and kernels](../kernels.md).

Learn how one weight row produces one output through a dot product.
Prerequisites: arrays, multiplication and addition. Follow the small example
below, then use the runnable [matrix-product benchmark](matrix-products.md)
to compare CPU implementations with .NET 10 and synthetic weights.

![MatVec: multiply each row of [[1, 2, -1], [0, 3, 2]] by the vector [2, 1, 4] and sum the products, producing [0, 11]](matvec.png)

[Open the MatVec illustration](matvec.png).

A **dot product** multiplies corresponding values and adds those products.
For the first weight row, `1 × 2 + 2 × 1 + (-1) × 4 = 0`; for the second,
`0 × 2 + 3 × 1 + 2 × 4 = 11`. Thus a `[2; 3]` weight matrix transforms a
three-value vector into two outputs, `[0; 11]`. Each output needs the whole
input vector, so the destination must use different storage from the input.
Each CPU worker processes one weight row at a time, reusing its temporary
integer or F32 row buffer. The default K-quant path is described in
[H.2](matrix-products.md#quantize-activations-once-for-k-quant-products).

Every projection of a block and the LM head multiply a stored weight matrix by
one F32 vector. `MatVec(weights, input, destination)` computes

```text
destination[r] = Σ weights[r, c] × input[c]
```

where `weights` is an `IWeightTensor` of `Rows` × `Columns` values, `input` has
shape `[Columns]` and `destination` has shape `[Rows]`. In the target model,
`attn_q` maps 3,072 values to 4,096, and the LM head maps them to the 131,072
logits of the vocabulary. Every tensor must belong to the executing backend.
`destination` must differ from `input`, because each output reads the whole
input. Validation and ordering otherwise follow `Add`.

The weights stay in their stored encoding. Core's `RowDecoder` decodes one row
of any `TensorEncoding` into F32, and `WeightMatrix.DecodeRow` applies it to a
row of a matrix. Block layouts and arithmetic follow `block_q4_0`, `block_q4_1`,
`block_q5_0`, `block_q5_1`, `block_q8_0`, `block_iq4_nl`, `block_q2_K`,
`block_q3_K`, `block_q4_K`, `block_q5_K` and `block_q6_K` in llama.cpp's
`ggml-common.h`, with the IQ4_NL levels of
`kvalues_iq4nl`, and its reference `dequantize_row_*` functions in
`ggml-quants.c`. Tests pack blocks of every quantized encoding by hand from
chosen scales, minimums and quants, and check every decoded value exactly. The
Q5_K and Q5_0 tensors of local `Q4_K_S` and `Q4_K_M` files decode to the same
mean, spread and range as Q4_K or Q8_0 tensors of the same roles, an optional
check; no local Q2_K or Q3_K file has been decoded yet. Embedding lookup uses
the same decoder.

In `DecodedFloat` mode, the CPU backend decodes one row at a time per worker into a pooled buffer.
`CpuMatrixKernels` uses safe `Vector256` or `Vector128` intrinsics when accelerated,
with a scalar fallback. Operands are widened before multiplication and summed
in four double-precision vector accumulators. Large products partition weight
rows between workers; each worker writes distinct output columns. Each output is rounded to F32 once. Because outputs are
sums of many products that can cancel, tests compare them with a tolerance
relative to the sum of the absolute products, not to the output itself.

On the local Ministral `Q4_K_M` file, decoded weights have a mean near zero, a
standard deviation of 0.004 to 0.012 and no NaN or infinity. Applying the LM head
to the normalized embedding of 10 sample tokens gives each token the highest
logit. On an Apple M4 Max, single-threaded in Release, a `Q4_K` product takes
about 0.5 ns per weight and a `Q6_K` product about 0.83 ns: 7 ms for `attn_q`,
336 ms for the LM head, and about 2.1 s for all matrix products of one token.
These are historical measurements before H.1. See the updated
[parallel matrix measurements](matrix-products.md#measure-parallel-matrix-products);
vectorizing the decoders remains separate work.
