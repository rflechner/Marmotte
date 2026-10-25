# Roadmap

Marmotte aims to run GGUF models locally from .NET. It currently reads GGUF
indexes, inspects them with the `info` command, and quantizes floating-point
weights to Q4_0 or Q8_0. See the
[implementation status](docs/gguf.md#marmotte-implementation-status) for details.

The first objective is to run a model on the CPU. CUDA and Metal backends follow
immediately after. The GGUF reader and quantizer continue to evolve alongside
these milestones.

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

### Weight access

- [x] Expose a format-neutral tensor source in `Marmotte.Core`, implemented by
  the GGUF and Safetensors libraries from their parsed descriptors, with row
  access to weight matrices. It copies validated payload ranges into managed
  arrays.
- [ ] Memory-map the GGUF file behind the same tensor source, after measuring
  load time and resident memory against the copying source.
- [x] Keep weights in their stored encoding: decoding the whole target model to
  F32 would need about 12.8 GiB.

### Tensor decoding

- [x] Decode F32, F16, BF16, Q4_0 and Q8_0 rows, so that synthetic models and the
  quantizer's output can run, in Core's `RowDecoder`, shared by embedding
  lookup and matrix products.
- [x] Decode Q4_K and Q6_K rows, required by the target model.
- [x] Verify block layouts against llama.cpp's reference implementation with
  independently encoded fixtures: `ggml-common.h` and `ggml-quants.c`, checked
  on 2026-10-25, with blocks packed by hand in `RowDecoderTests` and the
  embedding fixtures of `EmbeddingTests`.

### Model configuration

- [ ] Read architecture hyperparameters from typed metadata into an explicit
  configuration record, rejecting missing or inconsistent values.
- [ ] Map GGUF tensor names to transformer layers and check their shapes.

### Tokenizer

- [ ] BPE tokenizer built from `tokenizer.ggml.*` metadata: tokens, token types,
  merges, special token IDs and BOS/EOS flags.
- [ ] `tekken` pre-tokenization.
- [ ] Decode token IDs to UTF-8 text, including multi-byte sequences split across
  tokens during streaming.
- [ ] Chat prompts: start with raw prompts, then render `tokenizer.chat_template`
  (a Jinja template), or a supported subset of it.

### CPU kernels

The list follows llama.cpp's `mistral3` graph in
[`src/models/mistral3.cpp`](https://github.com/ggml-org/llama.cpp/blob/master/src/models/mistral3.cpp),
read on 2026-10-25, and the target model's metadata. For one token, the forward
pass looks up an embedding, runs 26 blocks of attention and feed-forward layers,
each followed by a residual connection, and ends with a final normalization and
the LM head. Sizes are rows × columns for one token; prompt processing runs the
same kernels over several tokens. Each generated token reads every weight once,
about 2 GiB, so the matrix products dominate the cost.

Contracts and element-wise operations:

- [x] Contiguous F32 execution tensors and an `Add` contract in Core, with
  explicit allocation/transfers and a `Vector<T>` CPU implementation in
  `Marmotte.Cpu`. `Add` provides the two residual connections of each block.
- [x] Embedding lookup: decode selected rows of `token_embd.weight` into F32,
  without scaling, through `IEmbeddingKernel`. CPU supports all seven runtime
  encodings, including the target model's `Q6_K` (131,072 rows of 3,072 values).
  Tests use small independent fixtures; the full model remains an optional check.
- [x] RMS normalization: `x / sqrt(mean(x²) + ε) × weight`, with ε from
  `attention.layer_norm_rms_epsilon` (1e-5) and F32 weights of 3,072 values.
  It runs before attention (`attn_norm`), before the feed-forward layer
  (`ffn_norm`) and before the LM head (`output_norm`): 53 times per token.
  `Marmotte.Cpu` vectorizes it with `Vector<T>` and sums the squares in double
  precision.
- [x] SwiGLU: `silu(gate) × up` over 9,216 values, with
  `silu(x) = x / (1 + e^-x)`, vectorized with `Vector<T>` and `Vector.Exp` in
  `Marmotte.Cpu`.

Matrix products:

- [x] Matrix-vector product over `Q4_K` and `Q6_K` rows, with F32 input and
  output: one dot product per weight row. `IMatVecKernel` in Core accepts every
  `TensorEncoding`; `Marmotte.Cpu` decodes one row at a time and sums in double
  precision. Each block uses seven:
  - `attn_q`: 4,096 × 3,072, `Q4_K`;
  - `attn_k` and `attn_v`: 1,024 × 3,072, `Q4_K`, except `attn_v` in `Q6_K` in
    half of the blocks;
  - `attn_output`: 3,072 × 4,096, `Q4_K`;
  - `ffn_gate` and `ffn_up`: 9,216 × 3,072, `Q4_K`;
  - `ffn_down`: 3,072 × 9,216, `Q4_K`, or `Q6_K` in half of the blocks.
- [x] LM head: the same product with `token_embd.weight` (`Q6_K`,
  131,072 × 3,072). The file has no `output.weight`: as in llama.cpp, the
  embeddings are tied and the LM head reuses the embedding matrix. Checked on
  the local file: each of 10 sample tokens gets the highest logit for its own
  normalized embedding.
- [x] Matrix-matrix products over the same weights, so that prompt processing
  handles several tokens per pass. `IMatMulKernel` takes `[tokens; columns]`
  inputs; `Marmotte.Cpu` decodes each weight row once for the whole batch,
  which makes 64-token batches about 3× faster per token than `MatVec`.

Attention:

- [x] RoPE on Q (32 heads) and K (8 heads), not V, over the full 128-value head.
  `mistral3` uses llama.cpp's "normal" variant, which rotates consecutive pairs
  of values (`LLAMA_ROPE_TYPE_NORM`), with base frequency 1,000,000 and the
  token's absolute position. `IRopeKernel` and `RotaryEmbedding` in Core support
  both pairings; `Marmotte.Cpu` computes the angles in double precision.
- [ ] YaRN frequency correction: scaling factor 16, original context 16,384,
  `beta_fast` 32 and `beta_slow` 1. With `yarn_log_multiplier` = 1, the
  attention magnitude correction computed in llama.cpp's `src/llama-context.cpp`
  should cancel out for this model. Implement the general formula, and check
  this case against ggml's RoPE implementation. `RotaryEmbedding.Create`
  accepts the corrected frequencies and magnitude.
- [ ] Attention temperature: after RoPE, multiply Q by
  `1 + β × ln(floor(p / L) + 1)`, where `p` is the token position,
  β = `attention.temperature_scale` (0.1) and
  L = `rope.scaling.original_context_length` (16,384). The factor is exactly 1
  for the first 16,384 positions. Source: `llm_graph_input_attn_temp` in
  llama.cpp's `src/llama-graph.cpp`.
- [ ] Key-value cache: append K, after RoPE, and V for each position and block,
  8 heads of 128 values each. That is 208 KiB per token in F32, or 832 MiB for
  a 4,096-token context; F16 halves it.
- [ ] Grouped-query attention: each of the 32 query heads reads the cached keys
  and values of its key-value head, shared by 4 query heads. Scores are scaled
  by 1/√128, masked causally, normalized with a numerically stable softmax
  (subtracting the maximum) and used to weight the cached values, with F32
  accumulation.

Not needed for the target model: the `mistral3` graph also handles biases and
mixture-of-experts layers, but the file has neither, and the architecture has
no Q/K normalization, sliding window, attention sinks, ALiBi or logit
soft-capping.

Optimizations, once reference implementations exist to compare against:

- [ ] Vectorize the matrix products with `System.Runtime.Intrinsics` and
  parallelize them across rows.
- [ ] Quantize the input vector once per product, as ggml's CPU backend does
  with `Q8_K` for K-quant dot products, instead of converting each weight block
  to F32.
- [ ] Compute the last block's feed-forward layer and the LM head only for the
  tokens whose logits are needed, and fuse attention into a single pass (flash
  attention).

### Inference and generation

- [ ] Forward pass for the `mistral3` architecture, structured so that other
  LLaMA-family architectures can follow.
- [ ] Prompt processing and token-by-token generation.
- [ ] Sampling: greedy first, then temperature, top-k and top-p with a seed.
- [ ] Stop on EOS or after a maximum number of tokens.

### CLI

- [ ] A command that generates text from a prompt and streams tokens to the
  console, for example `generate`.
- [ ] Report load time, prompt processing and generation throughput separately.

### Validation

- [ ] Test each kernel against a naive reference implementation.
- [ ] Run end-to-end tests with a tiny synthetic GGUF model.
- [ ] Compare with llama.cpp on the target model as an optional local check:
  token IDs for sample texts, and logits or top tokens for a fixed prompt.

## Milestone 2: CUDA and Metal backends

This milestone follows the CPU milestone directly. The CPU implementation remains
the reference for results.

- [x] Create `Marmotte.Cuda` and `Marmotte.Metal` projects implementing the common
  tensor and `Add` contracts with explicit `NotImplementedException` placeholders.
- [x] Define backend contracts for weights kept in their stored encoding,
  submission-ordered execution with explicit synchronization, and tensor
  lifetimes owned by the backend.
- [ ] Extend the boundary to the key-value cache and the operations used by the
  forward pass as they are implemented.
- [ ] Choose the interop approach for each backend: native kernels called through
  P/Invoke, or a .NET GPU library.
- [ ] CUDA backend: weight upload, quantized matrix products and the remaining
  kernels.
- [ ] Metal backend: Metal Shading Language kernels, sharing mapped weights
  through unified memory without copies where alignment allows.
- [ ] Compare each backend with the CPU reference within documented tolerances.
- [ ] Keep GPU tests optional, so that normal development does not require a GPU.
- [ ] Measure each backend with the same workloads as the CPU.

## GGUF reader and quantizer

These improvements continue alongside the milestones.

### Reader

- [ ] Optional lenient metadata key validation. The reader follows the key rules
  of the [GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md),
  dot-separated `lower_snake_case` segments, while llama.cpp only rejects empty
  and duplicate keys.
- [ ] Document the key validation rule in [docs/gguf.md](docs/gguf.md#metadata).
- [ ] Preserve the file's metadata key order alongside the `Metadata` map, and
  keep it when writing.

### Quantizer

- [ ] Accept `GgufReadOptions` to read the input and validate the output,
  accounting for the two metadata keys the quantizer adds.
- [ ] Write the padding after the last tensor, and test that offsets are
  contiguous in descriptor order, as llama.cpp's reader and writer expect.
- [ ] Tensor selection policies: keep the current shape-based policy and add a
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
- [ ] Report which tensors each policy keeps in floating point.
- [ ] K-quant encoders, starting with Q4_K and Q6_K, then mixed profiles such as
  `Q4_K_M`.
- [ ] Requantization from quantized inputs, using the decoders from milestone 1.

## Safetensors reader

- [ ] Sharded models: read `model.safetensors.index.json` and present the shards
  as one tensor index.
- [ ] Read companion files, such as `config.json` and tokenizer files, once
  Safetensors models become inference targets.
