# Choose the next token with temperature, top-k and top-p

After `GenerationSession.Prefill` or `Decode`, you have one logit per vocabulary
token. This walkthrough turns that row into a selected token. Prerequisites:
.NET 10, a built repository and the [generation session example](generation-session.md).
Sampling does not execute the model, decode text or decide when generation stops.

## 1. Run the example

```sh
dotnet build Marmotte.slnx
dotnet fsi examples/Generation.fsx -- --sample
```

Without `--sample`, the script still uses greedy. With it, the script creates
one sampler with temperature 0.8, top-k 3, top-p 0.9 and seed 42, then uses it
for all three selections. Repeating the command on the same runtime/platform
produces the same sequence. The model is synthetic: the printed IDs demonstrate
execution, not meaningful language. A selected token is added to the session
cache only when the caller passes it to `Decode`.

## 2. Configure one sampler per sequence

```fsharp
let parameters =
    TokenSamplingParameters.Create(temperature = 0.8, topK = Some 40, topP = 0.9)
let sampler = TokenSampler(parameters, seed = 42)

let chooseToken (logits: single array) =
    match sampler.Sample(ReadOnlySpan<single>(logits)) with
    | Ok tokenId -> tokenId
    | Error error -> invalidOp $"Token selection failed: {error}."
```

Keep this instance across generation steps. Recreating it with the same seed
before each token restarts the random stream. Parameters are immutable; independent
samplers can share them. Sampling instances are not thread-safe.

| Parameter | Meaning | Validation / disabling |
| --- | --- | --- |
| Temperature | Positive scale for relative logit differences; below 1 sharpens, above 1 flattens | Finite and > 0; use `TokenSampling.greedy` for deterministic maximum selection |
| TopK | Maximum number of eligible candidates | `Some k` with k > 0; `None` disables; k above the eligible count keeps all |
| TopP | Cumulative probability threshold after top-k and temperature | Finite in (0, 1]; 1 disables |
| Seed | Initializes this sampler's random stream | Any `int`; create a new instance to restart |

Invalid parameters raise `ArgumentOutOfRangeException` at construction. A null
parameter object raises `ArgumentNullException`. Parameters do not silently clamp
to defaults. Greedy remains an independent operation with no random state.

## 3. Understand the filtering order

The pipeline is:

```text
validate logits → rank → top-k → temperature + stable softmax → top-p → draw
```

Ranking uses descending logits and ascending token IDs for ties. Positive
temperature preserves this order, so top-k can run before scaling. Negative
infinity excludes a token. NaN or positive infinity anywhere is an error, even
if a filter would otherwise remove that entry.

After top-k, each retained token receives an unnormalized double-precision weight:

```text
weight[i] = exp((logit[i] - maximumRetainedLogit) / temperature)
```

Subtracting before dividing avoids positive overflow even for extremely small
temperatures. The maximum always has weight 1. Very negative relative scores
may underflow to zero; those entries are never selected.

Top-p operates on probabilities normalized over the top-k candidates. It keeps
the smallest non-empty prefix whose cumulative probability reaches p, including
the crossing token. Consider four equal logits, with top-k 3 and top-p 0.6:

| Token ID | Probability after top-k | Cumulative probability | Kept by top-p? |
| --- | --- | --- | --- |
| 0 | 1/3 | 1/3 | Yes |
| 1 | 1/3 | 2/3 | Yes: crosses 0.6 |
| 2 | 1/3 | 1 | No |
| 3 | Excluded | — | No |

The final draw therefore selects 0 or 1 with probability 1/2 each. With top-p
exactly 1/3, only token 0 is needed. With top-p 1, all three remain eligible.
A very small positive p always keeps at least the leading candidate.

Temperature also changes nucleus membership. For logits `[0; ln(4)]`,
probabilities at temperature 1 are `[0.2; 0.8]`; at temperature 2 they are
`[1/3; 2/3]`. With p = 0.75, the first distribution keeps only the leading token,
while the second must keep both. Greedy would select the leading token in both.

## 4. Reproducibility and failures

Each successful `Sample` consumes exactly one `System.Random.NextDouble` draw,
even if filtering leaves one candidate. The sampler draws against the retained
weight sum, equivalent to renormalizing that distribution. Invalid logits return
`TokenSamplingError` without consuming a draw, so a caller can correct input and
retry without shifting the sequence. Empty and entirely excluded vocabularies
return `EmptyLogits` and `NoEligibleTokens`, as with greedy.

Reproducibility requires the same seed, parameters, input logits, successful call
sequence, runtime and platform. This is not a cross-version random-stream or
floating-point compatibility guarantee. See the official
[System.Random reproducibility guidance](https://learn.microsoft.com/en-us/dotnet/fundamentals/runtime-libraries/system-random).
Different seeds can still select the same tokens; a seed does not force diversity.

Sampling reads but never mutates or retains the input span. Pass the complete
vocabulary row: returned IDs are indices relative to that span. EOS is an ordinary
candidate and does not stop the sampler. [`TokenGenerator`](token-generation.md)
applies the separate EOS and token-budget policy. Allocation failures propagate as exceptions.

## Validation and cost

The implementation sorts candidates, requiring O(V log V) time and O(V) scratch
for vocabulary size V. No performance optimization is claimed yet. Greedy retains
its O(V) scan and constant auxiliary space.

`TokenSamplingTests` checks analytically known distributions, exact nucleus
boundaries, crossing-token inclusion, filter ordering, deterministic ties,
reproducible streams, invalid input without state advancement, masked entries,
extreme temperatures/logits and input ownership. Distribution tests use a fixed
seed and broad tolerances; they require neither a model nor GPU hardware.
