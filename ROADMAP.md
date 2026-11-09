# Roadmap

Marmotte aims to run GGUF models locally from .NET. It currently reads GGUF
indexes, inspects them with the `info` command, and quantizes floating-point
weights to Q4_0 or Q8_0. See the
[implementation status](docs/gguf.md#marmotte-implementation-status) for details.

The first objective is to run a model on the CPU. CUDA and Metal backends follow
immediately after. The GGUF reader and quantizer continue to evolve alongside
these milestones.

Each group of checkboxes below has a letter (A, B, C, …), and its items are
numbered within the group: **A.1** is the first checkbox of group A, **A.2** the
second, and so on. Groups without checkboxes of their own, such as the
milestones, have no letter.

## Milestone 1: run a model on the CPU

The reference target is the Ministral 3 3B Instruct 2512 `Q4_K_M` model already
used by the launch profile and the exploration script. Its index defines what the
first implementation needs:

- Architecture `mistral3`: 26 blocks, embedding length 3072, feed-forward length
  9216, 32 attention heads and 8 key-value heads of 128 dimensions.
- RoPE with YaRN scaling, and an `attention.temperature_scale` parameter.
- 156 `Q4_K`, 27 `Q6_K` and 53 `F32` tensors, 2.00 GiB in total, with no
  `output.weight`: the LM head reuses the token embeddings.
- A `gpt2` BPE tokenizer with the `tekken` pre-tokenizer, 131,072 tokens and
  269,443 merges.

Automated tests use small synthetic models; the real model remains an optional
local check.

### A. Weight access

- [x] **A.1** Expose a format-neutral tensor source in `Marmotte.Core`, implemented by
  the GGUF and Safetensors libraries from their parsed descriptors, with row
  access to weight matrices. It copies validated payload ranges into managed
  arrays.
- [x] **A.2** Memory-map the GGUF file behind the same tensor source, after measuring
  load time and resident memory against the copying source. Implemented as
  `MappedTensorSource` in Core with `GgufTensorSource.openMappedFile`: zero-copy payload
  views over the mapping, no managed heap for weights. On a synthetic 2.0 GiB GGUF,
  reading all payloads by copy took 177 ms and +2.2 GiB of heap, while mapping served
  them in 60 ms with no heap growth; see
  [the measurement](docs/gguf.md#memory-mapped-tensor-source).
- [x] **A.3** Keep weights in their stored encoding: decoding the whole target model to
  F32 would need about 12.8 GiB.

### B. Tensor decoding

- [x] **B.1** Decode F32, F16, BF16, Q4_0 and Q8_0 rows, so that synthetic models and the
  quantizer's output can run, in Core's `RowDecoder`, shared by embedding
  lookup and matrix products.
- [x] **B.2** Decode Q4_K and Q6_K rows, required by the target model.
- [x] **B.3** Decode Q4_1, Q5_0, Q5_1 and IQ4_NL rows, added for the key-value cache;
  GGUF tensors in these encodings can also be read as weights.
- [x] **B.4** Decode Q2_K, Q3_K and Q5_K rows, so that the Q2_K and Q3_K_S/M/L
  profiles, which mix them with Q4_K and Q6_K, can be loaded. Layouts checked
  against the reference; local Q5_K tensors decode to the statistics of Q4_K
  tensors of the same roles. No Q2_K or Q3_K file has been decoded locally.
- [ ] **B.5** IQ2 and IQ3 encodings (`IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ3_XXS`,
  `IQ3_S`), which index fixed codebooks of 256 to 1,024 entries.
- [x] **B.6** Verify block layouts against llama.cpp's reference implementation with
  independently encoded fixtures: `ggml-common.h` and `ggml-quants.c`, checked
  on 2026-10-25, with blocks packed by hand in `RowDecoderTests` and the
  embedding fixtures of `EmbeddingTests`.

### C. Model configuration

- [x] **C.1** Read architecture hyperparameters from typed metadata into an explicit
  configuration record, rejecting missing or inconsistent values.
  `GgufModelConfiguration.read` supports dense `mistral3`, plain RoPE and YaRN;
  see the [supported contract](docs/model-configuration.md).
- [x] **C.2** Map GGUF tensor names to transformer layers and check their shapes.
  `GgufModelTensorLayout.read` validates the dense, bias-free `mistral3` layout,
  returns row-major descriptors by layer and reuses embeddings for absent output weights.

### D. Tokenizer

- [x] **D.1** BPE tokenizer built from `tokenizer.ggml.*` metadata: tokens, token types,
  merges, special token IDs and BOS/EOS flags.
  Implemented by Core's `BpeTokenizer` and `GgufTokenizer.read`, over explicitly
  pre-tokenized pieces. See the [tutorial and supported limits](docs/tokenizer.md).
- [x] **D.2** `tekken` pre-tokenization with Unicode scalar categories in
  `TekkenPreTokenizer.split`, used by `GgufTokenizer.EncodeText`. See the
  [tutorial, validation and compatibility limits](docs/tekken.md).
- [x] **D.3** Decode token IDs to UTF-8 text, including multi-byte sequences split across
  tokens during streaming. `BpeTokenizer.Decode` and `CreateDecoder` provide strict
  decoding with explicit special-token policy, completion and reset; see the
  [tutorial and error contracts](docs/token-decoding.md).
- [x] **D.4** Chat prompts: raw prompts and a supported subset of
  `tokenizer.chat_template`, through `JinjaRuntime`, `ChatPrompt.render` and
  `GgufTokenizer.RenderPrompt`. Produces text; special-token segmentation remains
  explicit. See [supported syntax, runtime limits and examples](docs/chat-prompts.md).

### CPU kernels

The list follows llama.cpp's `mistral3` graph in
[`src/models/mistral3.cpp`](https://github.com/ggml-org/llama.cpp/blob/master/src/models/mistral3.cpp),
read on 2026-10-25, and the target model's metadata. For one token, the forward
pass looks up an embedding, runs 26 blocks of attention and feed-forward layers,
each followed by a residual connection, and ends with a final normalization and
the LM head. Sizes are rows × columns for one token; prompt processing runs the
same kernels over several tokens. Each generated token reads every weight once,
about 2 GiB, so the matrix products dominate the cost.

**E. Contracts and element-wise operations:**

- [x] **E.1** Contiguous F32 execution tensors and an `Add` contract in Core, with
  explicit allocation/transfers and a `Vector<T>` CPU implementation in
  `Marmotte.Cpu`. `Add` provides the two residual connections of each block.
- [x] **E.2** Embedding lookup: decode selected rows of `token_embd.weight` into F32,
  without scaling, through `IEmbeddingKernel`. CPU supports every runtime
  encoding, including the target model's `Q6_K` (131,072 rows of 3,072 values).
  Tests use small independent fixtures; the full model remains an optional check.
- [x] **E.3** RMS normalization: `x / sqrt(mean(x²) + ε) × weight`, with ε from
  `attention.layer_norm_rms_epsilon` (1e-5) and F32 weights of 3,072 values.
  It runs before attention (`attn_norm`), before the feed-forward layer
  (`ffn_norm`) and before the LM head (`output_norm`): 53 times per token.
  `Marmotte.Cpu` vectorizes it with `Vector<T>` and sums the squares in double
  precision.
- [x] **E.4** SwiGLU: `silu(gate) × up` over 9,216 values, with
  `silu(x) = x / (1 + e^-x)`, vectorized with `Vector<T>` and `Vector.Exp` in
  `Marmotte.Cpu`.

**F. Matrix products:**

- [x] **F.1** Matrix-vector product over `Q4_K` and `Q6_K` rows, with F32 input and
  output: one dot product per weight row. `IMatVecKernel` in Core accepts every
  `TensorEncoding`; `Marmotte.Cpu` decodes one row at a time and sums in double
  precision. Each block uses seven:
  - `attn_q`: 4,096 × 3,072, `Q4_K`;
  - `attn_k` and `attn_v`: 1,024 × 3,072, `Q4_K`, except `attn_v` in `Q6_K` in
    half of the blocks;
  - `attn_output`: 3,072 × 4,096, `Q4_K`;
  - `ffn_gate` and `ffn_up`: 9,216 × 3,072, `Q4_K`;
  - `ffn_down`: 3,072 × 9,216, `Q4_K`, or `Q6_K` in half of the blocks.
- [x] **F.2** LM head: the same product with `token_embd.weight` (`Q6_K`,
  131,072 × 3,072). The file has no `output.weight`: as in llama.cpp, the
  embeddings are tied and the LM head reuses the embedding matrix. Checked on
  the local file: each of 10 sample tokens gets the highest logit for its own
  normalized embedding.
- [x] **F.3** Matrix-matrix products over the same weights, so that prompt processing
  handles several tokens per pass. `IMatMulKernel` takes `[tokens; columns]`
  inputs; `Marmotte.Cpu` decodes each weight row once for the whole batch,
  which makes 64-token batches about 3× faster per token than `MatVec`.

**G. Attention:**

- [x] **G.1** RoPE on Q (32 heads) and K (8 heads), not V, over the full 128-value head.
  `mistral3` uses llama.cpp's "normal" variant, which rotates consecutive pairs
  of values (`LLAMA_ROPE_TYPE_NORM`), with base frequency 1,000,000 and the
  token's absolute position. `IRopeKernel` and `RotaryEmbedding` in Core support
  both pairings; `Marmotte.Cpu` computes the angles in double precision.
- [x] **G.2** YaRN frequency correction: `RotaryEmbedding.Yarn` prepares frequencies
  and the final magnitude from `YarnParameters`. The target uses factor 16,
  original context 16,384, `beta_fast` 32 and `beta_slow` 1: pairs 20–37 form
  the transition. The general magnitude formula combines llama.cpp's context
  compensation with ggml's RoPE scaling; `yarn_log_multiplier` = 1 cancels
  the correction for this model. Checked against a compiled ggml CPU kernel
  and retained numerical fixtures, including position 262,143. See the
  [formulas and validation](docs/kernels.md#yarn-context-extension) and
  [runnable example](examples/Yarn.fsx).
- [x] **G.3** Attention temperature: after RoPE, multiply Q by
  `1 + β × ln(floor(p / L) + 1)`, where `p` is the token position,
  β = `attention.temperature_scale` (0.1) and
  L = `rope.scaling.original_context_length` (16,384). The factor is exactly 1
  for the first 16,384 positions. Source: `llm_graph_input_attn_temp` in
  llama.cpp's `src/llama-graph.cpp`. `AttentionTemperature` and
  `IAttentionTemperatureKernel` in Core; `Marmotte.Cpu` scales each token with
  `Vector<T>`.
- [x] **G.4** Key-value cache: append K, after RoPE, and V for each position and block,
  8 heads of 128 values each. That is 208 KiB per token in F32, or 832 MiB for
  a 4,096-token context; F16 halves it. `KeyValueCacheLayout`, `IKeyValueCache`
  and `IKeyValueCacheKernel` in Core give keys and values separate encodings,
  like LM Studio's K and V cache quantization: F32, F16, BF16, Q8_0 (221 MiB for
  4,096 tokens) or Q4_0 (117 MiB). Core's `RowEncoder` writes their rows, also
  for the GGUF quantizer. Storing at an earlier position replaces it and
  discards the later ones. `Marmotte.Cpu` allocates the whole capacity up front.
- [x] **G.5** Further cache encodings offered by llama.cpp: Q5_1 (156 MiB for 4,096
  tokens), Q5_0 (143 MiB), Q4_1 (130 MiB) and IQ4_NL (117 MiB, like Q4_0, but
  with levels closer together near zero). `TensorEncoding`, `RowDecoder` and
  `RowEncoder` represent them; IQ4_NL searches its scale by weighted least
  squares. Which pair of encodings the target model tolerates remains to be
  measured once attention runs.
- [x] **G.6** Grouped-query attention: each of the 32 query heads reads the cached keys
  and values of its key-value head, shared by 4 query heads, decoded from the
  cache's encoding one head at a time. Scores are scaled
  by 1/√128, masked causally, normalized with a numerically stable softmax
  (subtracting the maximum) and used to weight the cached values, with F32
  accumulation. `IAttentionKernel` and the CPU implementation support prompt batches
  and single-token decoding, with all nine cache encodings. See the
  [walkthrough](docs/kernels.md#grouped-query-attention) and `examples/Attention.fsx`.

Not needed for the target model: the `mistral3` graph also handles biases and
mixture-of-experts layers, but the file has neither, and the architecture has
no Q/K normalization, sliding window, attention sinks, ALiBi or logit
soft-capping.

**H. Optimizations, once reference implementations exist to compare against:**

- [x] **H.1** Matrix products use safe `Vector128`/`Vector256` intrinsics with
  a scalar fallback and parallel contiguous row partitions for large workloads.
  Double-precision arithmetic is preserved; see [measurements and validation](docs/kernels.md#measure-parallel-matrix-products).
- [x] **H.2** K-quant products quantize each input token once to transient Q8_K,
  share it across row workers and accumulate integer products with block scales.
  All five K encodings are supported; `DecodedFloat` preserves the H.1 reference
  policy. See [error bounds, fallback rules and measurements](docs/kernels.md#quantize-activations-once-for-k-quant-products).
- [x] **H.3** `LogitSelection` limits the final feed-forward and LM head to needed
  positions; generation skips intermediate batch logits and downloads only the
  last row. CPU online fused attention replaces the score buffer with running
  softmax/value sums. Full-output and reference-attention paths remain available;
  see [selected logits](docs/forward-pass.md#request-only-the-logits-you-need) and
  [attention measurements](docs/kernels.md#online-attention-and-h3-measurements).
  GPU tiling and sharing decoded K/V between query heads remain future work.

### I. Inference and generation

- [x] **I.1** Forward pass for the `mistral3` architecture, structured so that other
  LLaMA-family architectures can follow. Core's `TransformerModel` loads format-neutral
  weight roles and executes through `ITensorBackend`, with RoPE/YaRN, temperature,
  causal attention, both residuals, SwiGLU and tied or separate output. Synthetic
  tests compare logits with an independent reference; see the [walkthrough](docs/forward-pass.md).
- [x] **I.2** Prompt processing and token-by-token generation. `GenerationSession`
  owns a sequence cache, prefills prompts in bounded batches and consumes selected
  tokens one at a time, returning next-token logits. Includes reset, cancellation
  at batch boundaries and explicit fault handling; see the
  [walkthrough](docs/generation-session.md). Sampling and stopping policy remain separate.
- [x] **I.3** Sampling: `TokenSampling.greedy` provides deterministic maximum selection;
  `TokenSampler` applies top-k, temperature-scaled softmax and top-p with an explicit
  seed. Both report invalid logits explicitly. `examples/Generation.fsx --sample`
  demonstrates probabilistic selection; see the [sampling walkthrough](docs/sampling.md).
- [x] **I.4** `TokenGenerator` stops on explicit end-token IDs or a maximum number
  of new tokens, without decoding the final emitted token unnecessarily. It reports
  EOS, token/context limits and sampling failures, with per-token callbacks and
  cancellation; see the [walkthrough](docs/token-generation.md).

### J. CLI

- [x] **J.1** `generate` streams a greedy CPU continuation of a literal prompt from
  a dense `mistral3` GGUF with a `tekken` tokenizer. Includes EOS, token/context
  limits, incremental UTF-8 decoding and cancellation; see [Try a prompt](README.md#try-a-prompt).
- [x] **J.2** Report load time (as soon as loading finishes), prompt processing and
  generation throughput separately in the `generate` stderr output; see
  [Try a prompt](README.md#try-a-prompt).

### K. Validation

- [ ] **K.1** Test each kernel against a naive reference implementation.
- [ ] **K.2** Run end-to-end tests with a tiny synthetic GGUF model.
- [ ] **K.3** Compare with llama.cpp on the target model as an optional local check:
  token IDs for sample texts, and logits or top tokens for a fixed prompt.

## L. Milestone 2: CUDA and Metal backends

This milestone follows the CPU milestone directly. The CPU implementation remains
the reference for results.

- [x] **L.1** Create `Marmotte.Cuda` and `Marmotte.Metal` projects implementing the common
  tensor and `Add` contracts with explicit `NotImplementedException` placeholders.
- [x] **L.2** Define backend contracts for weights kept in their stored encoding,
  submission-ordered execution with explicit synchronization, and tensor
  lifetimes owned by the backend.
- [ ] **L.3** Extend the boundary to the key-value cache and the operations used by the
  forward pass as they are implemented.
- [ ] **L.4** Choose the interop approach for each backend: native kernels called through
  P/Invoke, or a .NET GPU library.
- [ ] **L.5** CUDA backend: weight upload, quantized matrix products and the remaining
  kernels.
- [ ] **L.6** Metal backend: Metal Shading Language kernels, sharing mapped weights
  through unified memory without copies where alignment allows.
- [ ] **L.7** Compare each backend with the CPU reference within documented tolerances.
- [ ] **L.8** Keep GPU tests optional, so that normal development does not require a GPU.
- [ ] **L.9** Measure each backend with the same workloads as the CPU.

## GGUF reader and quantizer

These improvements continue alongside the milestones.

### M. Reader

- [ ] **M.1** Optional lenient metadata key validation. The reader follows the key rules
  of the [GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md),
  dot-separated `lower_snake_case` segments, while llama.cpp only rejects empty
  and duplicate keys.
- [ ] **M.2** Document the key validation rule in [docs/gguf.md](docs/gguf.md#metadata).
- [ ] **M.3** Preserve the file's metadata key order alongside the `Metadata` map, and
  keep it when writing.

### N. Quantizer

- [ ] **N.1** Accept `GgufReadOptions` to read the input and validate the output,
  accounting for the two metadata keys the quantizer adds.
- [ ] **N.2** Write the padding after the last tensor, and test that offsets are
  contiguous in descriptor order, as llama.cpp's reader and writer expect.
- [ ] **N.3** Tensor selection policies: keep the current shape-based policy and add a
  llama.cpp-compatible one. As of
  [`src/llama-quant.cpp`](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-quant.cpp)
  on 2026-10-24, llama.cpp:
  - quantizes only tensors whose name ends with `weight`;
  - computes the rank like `ggml_n_dims`, ignoring trailing dimensions of 1;
  - keeps `ffn_gate_inp`, position embeddings, token types, `ssm_conv1d`,
    `shortconv.conv`, RWKV `time_mix_*`, `attn_rel_b`, vision and audio encoder
    tensors, and architecture-specific lists in floating point.

  Record the llama.cpp revision used, because these rules change with new
  architectures.
- [ ] **N.4** Report which tensors each policy keeps in floating point.
- [ ] **N.5** K-quant encoders, starting with Q4_K and Q6_K, then mixed profiles such as
  `Q4_K_M`.
- [ ] **N.6** Requantization from quantized inputs, using the decoders from milestone 1.

## O. Safetensors reader

- [ ] **O.1** Sharded models: read `model.safetensors.index.json` and present the shards
  as one tensor index.
- [ ] **O.2** Read companion files, such as `config.json` and tokenizer files, once
  Safetensors models become inference targets.

## P. Declarative forward architectures

Explore describing forward passes as data, so that a model variant composed of
existing operations can be supported without another hand-written orchestration.
Kernels remain implemented in F# through `ITensorBackend`; a genuinely new
operation still requires implementation and tests. Start with the typed graph
before designing its YAML representation.

- [ ] **P.1** Define a format-neutral, typed F# graph for forward operations, weight
  roles, configuration parameters, intermediate tensors, repeated blocks and
  explicit key-value cache reads/writes. Reproduce the current dense `mistral3`
  forward and compare logits with the existing implementation and independent
  numerical reference.
- [ ] **P.2** Validate operation support, weight roles, tensor dimensions and
  dependencies before loading weight payloads or submitting computation. Report
  errors with the affected operation and value; do not infer missing semantics
  from weight names or shapes.
- [ ] **P.3** Build an execution plan over `ITensorBackend`, tracking intermediate
  lifetimes and reusing buffers where safe, while preserving residual inputs,
  cache effects and backend submission order.
- [ ] **P.4** Add a versioned YAML schema that constructs the validated graph,
  including configuration references and repeated transformer blocks. Keep YAML
  parsing separate from execution, reject unknown operations and fields, and
  document the supported vocabulary with runnable examples.
- [ ] **P.5** Demonstrate another architecture variant expressed with existing
  operations, with numerical regression tests. Document which changes need only
  a description and which require new kernels or model-format adapters.
