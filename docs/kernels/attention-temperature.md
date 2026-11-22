# Attention Temperature

[Back to execution tensors and kernels](../kernels.md).

Learn how a token position selects one factor for all of its query values.
Prerequisites: the [RoPE walkthrough](rope.md), multiplication and the natural
logarithm. The worked boundary values below show how the factor changes
without mixing values from different tokens.

`mistral3` also multiplies its queries, after RoPE, by a factor that grows with
the position:

```text
factor(p) = 1 + β × ln(floor(p / L) + 1)
```

`β` is `attention.temperature_scale` and `L` is
`rope.scaling.original_context_length`: 0.1 and 16,384 for the target model.
The factor is exactly 1 for the first 16,384 positions, then stays constant over
each block of 16,384 positions: about 1.0693 (`1 + 0.1 × ln 2`) from position
16,384, 1.1099 from 32,768, and 1.2773 at position 262,143, the last position of
the training context. Multiplying a query multiplies all its attention scores by
the same factor, which sharpens their softmax over long contexts. Keys and
values are not scaled. This is unrelated to the sampling temperature of
generated tokens.

`AttentionTemperature.Create(float 0.1f, 16384)` holds these parameters, and
`Factor(position)` returns the F32 factor of a position.
`ApplyAttentionTemperature(queries, positions, temperature, destination)`
multiplies every value of each token by the factor of its position. `queries`
has rank 2 or more, with one token per row of the first dimension, as for
`Rope`. `destination` may be `queries`, and a negative position raises
`ArgumentOutOfRangeException` before any write.

```fsharp
let temperature = AttentionTemperature.Create(float 0.1f, 16384)

backend.Rope(queries, positions, embedding, queries)
backend.ApplyAttentionTemperature(queries, positions, temperature, queries)
backend.Rope(keys, positions, embedding, keys)
```

The formula follows `llm_graph_input_attn_temp` in llama.cpp's
`src/llama-graph.cpp`, which evaluates it in double precision and stores it in
F32; its `mistral3` graph then multiplies the queries after RoPE. Marmotte
computes the floor with an exact integer division. llama.cpp's position offset,
zero for `mistral3`, and its power variant are not supported. The CPU backend
computes one factor per token and multiplies with `Vector<T>`. Each value is
multiplied once, so positions below `L` leave queries unchanged, bit for bit.
Tests check the factors at block boundaries against values computed by hand,
the per-token scaling in rank 2 and rank 3, in-place use, empty batches and
validation.
