# Matrix-Matrix Product

[Back to execution tensors and kernels](../kernels.md).

Learn how a batch of tokens shares the same weight matrix. Start with the
[MatVec walkthrough](matvec.md), then follow the two-token example below.
The runnable [matrix-product benchmark](matrix-products.md) uses .NET 10
and synthetic weights, without a model download.

![MatMul: apply the same two weight rows to token inputs [2, 1, 4] and [1, 0, -1], producing rows [0, 11] and [2, -2]; decode each weight row once for both tokens](matmul.png)

[Open the MatMul illustration](matmul.png).

Reuse the [MatVec](matvec.md) weights, but provide two token rows: `[2; 1; 4]` and
`[1; 0; -1]`. Their outputs are `[0; 11]` and `[2; -2]`. Tokens remain
independent: the kernel sums across feature columns, not across tokens.

In `DecodedFloat` mode, the CPU's outer loop decodes one weight row; its inner loop
uses that row for every token. Weight row `0` therefore fills output column `0`
with `[0; 2]`. Weight row `1` fills column `1` with `[11; -2]`. This is why
`Y = X × Wᵀ` has shape `[tokens; output features]`. The transpose describes
the mathematical relationship; the implementation reads stored rows directly
without creating a transposed copy.

Prompt processing runs the same projections for every token of the prompt.
`MatMul(weights, input, destination)` multiplies a batch of token rows by the
same stored weights:

```text
destination[t, r] = Σ weights[r, c] × input[t, c]
```

`input` has shape `[tokens; Columns]` and `destination` `[tokens; Rows]`: each
token's row of `input` gives its row of `destination`, as `MatVec` would for that
token alone. In matrix terms, this multiplies `input` by the transposed weight
matrix, because weight rows are the outputs. A batch of zero tokens is a valid
no-op. `destination` must differ from `input`. Ownership, validation, ordering
and tolerances follow `MatVec`.

The row preparation cost is what a batch saves. The CPU backend decodes or
unpacks each weight row once into its buffer, then sums its products with the
input row of every token. Each output is computed exactly as `MatVec` computes it, so batched and
token-by-token results are identical on the CPU; a test checks this. `MatVec`
itself is the one-token case of the same kernel.

Before H.1, on the local Ministral file, single-threaded in Release on an Apple M4 Max:

| Matrix | `MatVec` | 4 tokens | 16 tokens | 64 tokens |
| --- | --- | --- | --- | --- |
| `attn_q` (`Q4_K`, 4,096 × 3,072) | 7.4 ms | 2.8 ms per token | 2.2 ms per token | 2.05 ms per token |
| `ffn_down` (`Q6_K`, 3,072 × 9,216) | 24.6 ms | 9.8 ms per token | 5.8 ms per token | 4.9 ms per token |

With 64-token batches, all matrix products of the model take about 0.57 s per
prompt token, against about 1.8 s one token at a time in the same run. Beyond
about 16 tokens, the double-precision sums of products dominate, at about
0.17 ns per weight and token.

See [CPU matrix products](matrix-products.md) for current arithmetic policies,
parallel scheduling and the later H.1/H.2 measurements.
