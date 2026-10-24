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
- 156 `Q4_K`, 27 `Q6_K` and 53 `F32` tensors, 2.00 GiB in total.
- A `gpt2` BPE tokenizer with the `tekken` pre-tokenizer, 131,072 tokens and
  269,443 merges.

Automated tests use small synthetic models; the real model remains an optional
local check.

### Weight access

- [ ] Memory-map the GGUF file and expose read-only tensor views over validated
  payload ranges, reusing the parsed descriptors.
- [ ] Keep weights in their stored encoding: decoding the whole target model to
  F32 would need about 12.8 GiB.

### Tensor decoding

- [ ] Decode F32, F16, BF16, Q4_0 and Q8_0 rows, so that synthetic models and the
  quantizer's output can run.
- [ ] Decode Q4_K and Q6_K rows, required by the target model.
- [ ] Verify block layouts against llama.cpp's reference implementation with
  independently encoded fixtures.

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

- [ ] Matrix-vector and matrix-matrix products over quantized weights.
- [ ] RMS normalization, RoPE with YaRN scaling, softmax, SwiGLU and embedding
  lookup.
- [ ] Grouped-query attention with a key-value cache.
- [ ] Vectorize with `System.Runtime.Intrinsics` and parallelize across rows,
  once a straightforward reference implementation exists to compare against.

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

- [ ] Extract a backend boundary from the working CPU implementation: weight
  storage, key-value cache, and the operations used by the forward pass.
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
