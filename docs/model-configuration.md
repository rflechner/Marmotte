# Model configuration

`GgufModelConfiguration.read model.Metadata` interprets metadata separately from
`Gguf.read`. It performs no I/O and allocates no buffers proportional to tensor
or context sizes. The result is Core's format-neutral `ModelConfiguration`
record. Invalid or unsupported metadata raises `InvalidDataException`, with the
relevant metadata key. Unknown architectures remain readable by the index reader.

The initial scope is dense `mistral3` with scalar layer dimensions, full attention
and full-head adjacent-pair RoPE, either unscaled or YaRN. The configuration reader does not check tensor names or shapes; the separate
layout reader described below checks them. Tokenizer validation and execution
remain separate steps. This configuration alone does not establish that a model
can execute.
Other architectures, per-layer arrays, mixture of experts, nonzero sliding
windows, partial rotary dimensions and other scaling modes are rejected.

## Typed metadata contract

All keys below have the `mistral3.` prefix, except `general.architecture`, which
must be the String `mistral3`. Counts must be positive UInt32 values fitting in
Int32, and floating-point parameters must be finite Float32 values. Other numeric
types are rejected rather than coerced. Extra unrelated metadata is ignored.

Required counts are `context_length`, `block_count`, `embedding_length`,
`feed_forward_length`, `attention.head_count`, `attention.head_count_kv`,
`attention.key_length` and `attention.value_length`. Explicit head lengths avoid
assuming that the embedding width equals the query projection width. The target
has embedding width 3072 but 32 query heads of 128 values (width 4096).
The key-value head count must divide the query head count, and projected widths
must fit Int32. Required `attention.layer_norm_rms_epsilon` and `rope.freq_base`
are positive. YaRN additionally requires a frequency base greater than one.

| Optional key | Default / constraint |
| --- | --- |
| `rope.dimension_count` | Key length; must equal it, and be even |
| `rope.scaling.type` | `none`; only `none` and `yarn` are supported |
| `attention.temperature_scale` | 0; non-negative |
| `expert_count`, `expert_used_count`, `attention.sliding_window` | Absent or UInt32 zero |

For YaRN, `rope.scaling.factor` is required and at least 1.
`rope.scaling.original_context_length` is required for YaRN or nonzero attention
temperature, must be positive and cannot exceed `context_length` when used.
The original context is a count of token positions, not an execution context
allocation. YaRN's optional Float32 values are:

| Key | Default / constraint |
| --- | --- |
| `rope.scaling.yarn_beta_fast` | 32; at least beta_slow |
| `rope.scaling.yarn_beta_slow` | 1; positive |
| `rope.scaling.yarn_log_multiplier` | 0; non-negative |
| `rope.scaling.attn_factor` | 1; positive |

The effective rotary magnitude must remain positive and finite in F32.
Scaling parameters without `yarn` are rejected to avoid silently discarding them.
These are explicit interpretation defaults; they are not inserted into the
original metadata map. The public Core record can also be constructed manually
and does not enforce validation at construction.

## Use

```fsharp
open Marmotte.Core
open Marmotte.Gguf

let index = Gguf.readFile "model.gguf"
let configuration = GgufModelConfiguration.read index.Metadata
let rotaryEmbedding =
    match configuration.RopeScaling with
    | ModelRopeScaling.None ->
        RotaryEmbedding.Standard(configuration.KeyLength,
                                 configuration.RopeFrequencyBase,
                                 RopePairing.AdjacentPairs)
    | ModelRopeScaling.Yarn parameters ->
        RotaryEmbedding.Yarn(configuration.KeyLength,
                             configuration.RopeFrequencyBase,
                             RopePairing.AdjacentPairs, parameters)
```

## References and verification

Metadata names, types and head dimensions were checked against the
[GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
and the [mistral3 model definition](https://github.com/ggml-org/llama.cpp/blob/master/src/models/mistral3.cpp)
on 2026-11-08. The implementation is original F#; its deliberately restricted
acceptance policy is described above. YaRN defaults and magnitude follow the
[existing kernel contract and reference checks](kernels.md#yarn-context-extension).

`ModelConfigurationTests` uses small typed metadata maps, including the target's
hyperparameters, and checks defaults, missing keys, exact types, invalid numbers,
inconsistent dimensions and unsupported variants. No model installation is
required. Reading configuration does not change stream ownership or position.

## Tensor roles and shape validation

See the [illustrated guide to TransformerLayerTensors](transformer-layer-tensors.md)
for a walkthrough of the nine weights inside one block.

`GgufModelTensorLayout.read index` reads C.1 configuration and maps an existing
GGUF index into Core's `ModelTensorLayout`. It never opens a file or reads weight
payloads. The layer collection is read-only and ordered by zero-based block
index, independently of the descriptor order in the file. Each descriptor has a
new shape array in row-major order; callers must treat these arrays as read-only.

The first supported layout is dense, bias-free `mistral3`, with separate query,
key and value matrices and equal key/value head dimensions. Let E be embedding
length, F feed-forward length, Q query head count times key length, K key-value
head count times key length, and V the number of embedding rows. V is explicitly
derived from `token_embd.weight`; it is not a validated tokenizer vocabulary size.
It must be positive and fit Int32. The accepted shapes are:

| GGUF name | Core role | Row-major shape |
| --- | --- | --- |
| `token_embd.weight` | TokenEmbedding | [V; E] |
| `output_norm.weight` | OutputNormalization | [E] |
| `output.weight` | Output | [V; E] |
| `blk.{i}.attn_norm.weight` | AttentionNormalization | [E] |
| `blk.{i}.attn_q.weight` | Query | [Q; E] |
| `blk.{i}.attn_k.weight` | Key | [K; E] |
| `blk.{i}.attn_v.weight` | Value | [K; E] |
| `blk.{i}.attn_output.weight` | AttentionOutput | [E; Q] |
| `blk.{i}.ffn_norm.weight` | FeedForwardNormalization | [E] |
| `blk.{i}.ffn_gate.weight` | FeedForwardGate | [F; E] |
| `blk.{i}.ffn_up.weight` | FeedForwardUp | [F; E] |
| `blk.{i}.ffn_down.weight` | FeedForwardDown | [E; F] |

All weights except `output.weight` are required, with exact ranks and dimensions.
If output is absent, `HasTiedOutput` is true and `Output` references the same
descriptor as `TokenEmbedding`. An explicit output must match the embedding shape.
No payload is duplicated or inferred from a filename.

Missing, duplicate and unexpected tensor names and incompatible shapes raise
`InvalidDataException`. Unexpected tensors are rejected so unsupported biases,
expert weights, additional rotary factors and extra blocks cannot be silently
omitted. This strict policy may reject valid GGUF variants outside the initial
scope. A tensor-count check bounds layer allocation before using block_count.

Encodings and byte sizes are preserved, including unknown values: successful
shape mapping does not promise that a backend can execute the weights. Payload
bounds and index integrity remain the GGUF reader's responsibility, and runtime
sources separately enforce little-endian, single-file payload support. The layout
is independent of streams and backends and changes no resource ownership.

```fsharp
let layout = GgufModelTensorLayout.read index
let firstQuery = layout.Layers[0].Query
printfn "%s: %A" firstQuery.Name firstQuery.Shape
printfn "Vocabulary rows: %d; tied output: %b" layout.VocabularySize layout.HasTiedOutput
```

Names, dimensions, output fallback and equal head lengths were verified against
the [mistral3 tensor and graph definitions](https://github.com/ggml-org/llama.cpp/blob/master/src/models/mistral3.cpp)
on 2026-11-08. `ModelTensorLayoutTests` uses independent synthetic index descriptors
with different embedding/query widths and shuffled blocks. Tests cover each role,
wrong shapes, missing and extra names, explicit and tied outputs, bounded allocation,
shape ownership and unknown encodings. They require no local model or weight I/O.
