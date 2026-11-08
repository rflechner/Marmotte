# Model configuration

`GgufModelConfiguration.read model.Metadata` interprets metadata separately from
`Gguf.read`. It performs no I/O and allocates no buffers proportional to tensor
or context sizes. The result is Core's format-neutral `ModelConfiguration`
record. Invalid or unsupported metadata raises `InvalidDataException`, with the
relevant metadata key. Unknown architectures remain readable by the index reader.

The initial scope is dense `mistral3` with scalar layer dimensions, full attention
and full-head adjacent-pair RoPE, either unscaled or YaRN. Tensor names, shapes,
biases, tied embeddings, vocabulary and tokenizer validation belong to later
steps; this configuration alone does not establish that a model can execute.
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
