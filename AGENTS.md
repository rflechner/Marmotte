# Coding guidelines for Marmotte

These guidelines apply to all code written or modified in this repository.
They adapt the principles of [RuleMaskDb's coding guidelines](https://github.com/rflechner/RuleMaskDb/blob/develop/AGENTS.md)
to F#, .NET, and local model inference.

## Use English and descriptive names

- Write identifiers, code comments, API documentation, and new developer documentation in English.
- Use the language requested by the user for conversation and user-facing output.
- Choose names that explain intent. Keep established format names such as GGUF, GGML, and `Q3_K` when they are more precise than a paraphrase.
- Use PascalCase for types, modules, union cases, and members; use camelCase for functions, parameters, and local bindings.
- Keep changes focused. Do not translate or reformat unrelated existing files to apply these guidelines retroactively.

## Understand the project boundaries

- Marmotte is an F# solution targeting .NET 10. Check the project files before changing framework or dependency assumptions.
- Manage explicit NuGet package versions centrally in the root `Directory.Packages.props`. Add `PackageReference` entries without `Version` attributes to the projects that need them.
- Include an explicit `FSharp.Core` package reference in each F# project: Central Package Management disables the SDK's implicit reference. Keep its version in `Directory.Packages.props` too.
- `src/Marmotte.Gguf` owns GGUF types, index reading, metadata, tensor descriptors, and native streaming quantization. `Gguf.read` remains an index-only operation; `GgufQuantizer` separately converts F32/F16/BF16 weights to Q4_0 or Q8_0 with Core's `RowEncoder`, and `GgufTensorSource` maps the index to Core's tensor source. Core's `RowDecoder` dequantizes rows.
- `src/Marmotte.Safetensors` owns Safetensors types and index reading for a single file. `Safetensors.read` validates the JSON header, tensor descriptors, and exact data coverage without reading weights; `SafetensorsTensorSource` maps the index to Core's tensor source. Sharded indexes are not implemented.
- `src/Marmotte.Core` contains shared helpers and model and inference domain code. It owns format-neutral tensor access: `TensorEncoding`, `TensorDescriptor`, `ITensorSource`, the copying `FileTensorSource`, and `WeightMatrix`. It also defines execution contracts: `TensorShape`, `ITensor`, `IWeightTensor`, `IKeyValueCache`, `IAddKernel`, `ISwiGluKernel`, `IRmsNormKernel`, `IEmbeddingKernel`, `IMatVecKernel`, `IMatMulKernel`, `IRopeKernel`, `IAttentionTemperatureKernel`, `IKeyValueCacheKernel`, and `ITensorBackend`, with `RotaryEmbedding` for RoPE, `YarnParameters` for preparing YaRN frequencies and magnitude, `AttentionTemperature` for the position-dependent query factor, and `KeyValueCacheLayout` for the dimensions and separate key and value encodings of a cache. `RowDecoder` decodes rows of every `TensorEncoding` into F32; `RowEncoder` writes F32, F16, BF16, Q8_0 and Q4_0 rows and stores unrepresentable quantized blocks with a NaN scale. Core references no file format or backend; their libraries implement the contracts. Memory mapping is not implemented yet.
- `src/Marmotte.Cpu` implements owned contiguous F32 and F16 tensors, explicit copying transfers, weight tensors that retain the stored payload, safe `Vector<T>` F32 `Add`, scalar Half `Add`, SwiGLU and RMS normalization kernels, embedding lookup, `MatVec` and `MatMul`. All decode weight rows with Core's `RowDecoder`, for all seven runtime encodings: the lookup writes selected rows directly into its destination, `MatVec` decodes one row at a time and multiplies it by an F32 vector, and `MatMul` shares each decoded row between every token of a batch. RoPE computes its angles in double precision once per token, and attention temperature one F32 factor per token. The key-value cache allocates its whole capacity up front and encodes and decodes batches of positions with Core's `RowEncoder` and `RowDecoder`; each layer stores contiguous positions, and storing at an earlier position discards the later ones. `CpuKernels.fs` holds the arithmetic over spans, without tensors or ownership; `CpuBackend.fs` holds the storage, ownership, lifetime and argument validation, and calls those kernels. All operations complete before returning. Embedding lookup validates every token ID before writing and decodes only selected rows into a `[token count; columns]` F32 destination, without scaling. Tensors belong to their creating backend instance, which releases those still alive when it is disposed. `src/Marmotte.Cuda` implements owned F32 and F16 device tensors, zero-initialized allocation, copying transfers, synchronization and Add through a native CUDA library. `CudaBackend.fs` owns validation and lifetimes; `CudaNative.fs` owns safe P/Invoke; `native/CudaRuntime.cu` owns streams and memory, and each operation has one `.cu` file under `native/kernels/` containing private typed F32 (`float`) and F16 (`half`) GPU functions and a shared host entry point that validates and dispatches by allocation encoding. List sources in the single `native/CMakeLists.txt`; split an operation into more files only when its complexity justifies it. Add implements both variants. F16 and Half name the same format: CPU owns `Half[]`, CUDA allocates two bytes per element, and transfers preserve raw bits without an F32 staging buffer. `Allocate(shape)` stays F32; `Allocate(shape, encoding)` supports F32/F16, and typed `Upload`/`Download` overloads use `single` or `Half`. Add rejects mixed formats; other CPU kernels reject F16 before writing. Follow `native/kernels/README.md` when adding an operation. Native builds are opt-in through `MarmotteBuildCuda=true`, which invokes CMake before the F# build and propagates the native library to consumers; prebuilt libraries use `MarmotteCudaNativePath` instead. GPU tests are opt-in too; see `docs/cuda.md`. Other CUDA operations and all `src/Marmotte.Metal` operations raise `NotImplementedException`. Backend operations take effect in submission order and may complete asynchronously; `Download` and `Synchronize` wait for them. Host token IDs must be consumed or copied before lookup returns.
- `src/Marmotte.Cli` owns Spectre.Console orchestration and presentation, including `info` for inspecting GGUF and Safetensors files, detected from their content, and `quantize` for writing a new GGUF. There is no inference pipeline yet.
- Keep console output and application-specific decisions in the CLI or exploration scripts. Libraries should expose usable data and explicit contracts.
- Reuse `src/Marmotte.Cli/ConsoleDsl.fs` for table construction, text previews, byte formatting, and section rendering. Keep format-specific interpretation in `GgufReport.fs` and `SafetensorsReport.fs`.
- `tests/Marmotte.Gguf.Tests` and `tests/Marmotte.Safetensors.Tests` contain xUnit tests using small binary fixtures. `tests/Marmotte.Core.Tests` tests the tensor types with an in-memory source and temporary files, row decoding against blocks packed by hand, row encoding against hand-computed bytes, and key-value cache layouts. Keep automated tests independent of local model installations.
- `tests/Marmotte.Cli.Tests` runs the CLI against temporary GGUF and Safetensors fixtures to check output, argument handling, and exit codes. Treat model text as literal console content, and keep metadata and tensor previews bounded.
- `tests/Marmotte.Backends.Tests` covers CPU Add, SwiGLU, RMS normalization, embedding lookup, MatVec, MatMul, RoPE, attention temperature and the key-value cache, independently encoded row fixtures, token IDs, transfers, shape compatibility, ownership and disposal, plus failures for unfinished GPU operations. Shared CPU/CUDA Half tests cover independent binary16 rounding fixtures and every transfer bit pattern. Optional CUDA Add tests cover F32/F16 arithmetic, transfers, ordering, ownership and disposal on a real GPU; default tests need no CUDA installation. YaRN tests cover Core parameter preparation and CPU rotation, including recorded ggml outputs without a native dependency. Keep these tests independent of GPU hardware and drivers. `examples/Add.fsx`, `examples/Embedding.fsx`, `examples/Yarn.fsx` and `examples/KeyValueCache.fsx` demonstrate the backend contracts without a model file; `examples/CudaAdd.fsx` exercises CUDA after the optional native build.
- Keep dependencies directed toward reusable libraries. Do not make libraries depend on the CLI or scripts, or introduce architectural layers without a concrete need.

## Write original code

- Never copy code from another project, especially well-known ones such as llama.cpp, ggml, Hugging Face Transformers, or the Safetensors reference implementation. This applies to library code, kernels, shaders, tests, and scripts, and also to translating such code line by line into F#.
- Use other projects only as references. Read their specifications, documentation, and implementations to understand and verify behavior, such as binary layouts, constants, formulas, and expected values. Then write Marmotte's implementation with its own structure and idioms, and cite the source as described in [Document contracts and intent](#document-contracts-and-intent).

## Document contracts and intent

- Use F# `///` documentation for public APIs, important contracts, and non-obvious behavior.
- Explain why a decision exists, including constraints and tradeoffs, rather than restating the implementation.
- Document units, dimension order, supported versions, stream requirements, stream ownership, final stream position, and exception behavior where relevant.
- Make partial knowledge visible: an unknown tensor encoding has an unknown byte size, not a zero-byte payload; a file profile does not establish every tensor's encoding.
- Update the README, examples, and API documentation when a behavior or public contract changes.
- Verify binary format tags and layouts against the upstream specification or implementation. Record the relevant source in the documentation under `docs/` instead of relying on remembered constants.
- Keep the names of reference projects out of the code. Comments, API documentation, test names, and scripts do not mention llama.cpp, its ggml library, or LM Studio, nor their URLs, source paths, or function names. Describe the behavior itself and point to the documentation that cites the source, such as "the reference quantizers cited in docs/gguf.md". Format and domain vocabulary such as GGUF, GGML, `GgmlType`, and metadata keys is not affected, nor are the `LM_STUDIO_MODELS_FOLDER` variable and model paths. The README, the roadmap, and `docs/` may name these projects.

## Prefer readable, idiomatic F#

- Favor straightforward code over clever, dense expressions. Give functions and modules one cohesive responsibility.
- Prefer immutable bindings and pure transformations. Keep I/O and resource management at clear boundaries.
- Use records for related data and discriminated unions for alternatives with different meanings. Prefer exhaustive pattern matching to flags, casts, or loosely typed dictionaries.
- Use `option` for meaningful absence and `Result` for expected failures when designing a new API. Preserve established exception-based contracts where they are appropriate: the GGUF and Safetensors readers report malformed files with `InvalidDataException` and let ordinary I/O failures propagate.
- Use classes and interfaces when they clarify resource ownership, mutable state, .NET interoperation, or a real dependency boundary. Apply SOLID pragmatically; do not mechanically translate object-oriented patterns into F#.
- Local mutation, arrays, spans, and explicit loops are appropriate for binary reading and measured performance needs. Keep mutation scoped and avoid shared mutable state.
- A record containing arrays is not deeply immutable. Consider ownership and copying when exposing mutable buffers, especially large ones.
- Add type annotations where they clarify public contracts, numeric widths, units, or overload resolution. Avoid unnecessary annotations on obvious local expressions.
- Respect F# compilation order: list dependencies before their consumers in `.fsproj` files and `#load` directives.

## Express domain concepts explicitly

- Use the domain's vocabulary consistently: metadata value, tensor descriptor, quantization block, file profile, alignment, and data offset.
- Represent meaningful concepts with focused records, unions, or small types when they remove ambiguity. Avoid long tuples or parameter lists whose meaning callers must reconstruct.
- Keep invariants and interpretation in the type or module that owns the concept. Expose useful members or functions rather than repeating interpretation logic in callers.
- Keep `GgmlType` and `GgufFileType` separate. `Q3_K` is a tensor encoding; `Q3_K_S`, `Q3_K_M`, and `Q3_K_L` are file-level profiles. Their numeric identifiers belong to different domains.
- Preserve the distinction between offsets relative to the tensor data section and absolute positions in the source stream. Preserve GGUF dimensions in GGML order, with the contiguous dimension first, and Safetensors shapes in row-major order, with the contiguous dimension last. Core's `TensorDescriptor.Shape` is row-major; only the GGUF mapping reverses dimensions, without moving bytes.
- Preserve unknown tensor and file-profile identifiers where the format remains readable, and state which validations are then unavailable. Reject unknown metadata tags when their payload layout cannot be determined.
- Do not invent metadata or infer facts from filenames. Any deliberate inference should be explicit in the API and distinguishable from values read from the file.

## Keep functions focused and give code breathing room

- Split long functions at meaningful responsibility boundaries into named helpers. Avoid arbitrary line-count rules and helpers that merely obscure a single expression.
- Separate validation, reading, interpretation, and presentation when doing so makes the flow clearer.
- Prefer explicit pattern matches and early validation over deeply nested conditions. Keep pipelines short enough to understand without reconstructing several intermediate types.
- Use four-space indentation and follow the surrounding F# formatting.
- Put distinct logical steps on separate lines with appropriate blank lines. Avoid multiple statements on one line and long, compressed record construction or pipeline expressions.

## Treat binary input and large models carefully

- Treat file contents as untrusted data. Validate magic, version, tags, lengths, counts, dimensions, alignment, and offsets before using them.
- Check bounds and arithmetic overflow before allocating buffers, multiplying sizes, adding offsets, or narrowing integer types. Use explicit integer widths and byte order.
- Enforce the reader's configurable allocation and nesting limits. Do not weaken them merely to make a malformed fixture or unsupported model pass.
- Handle partial stream reads and truncated files. Preserve diagnostic byte positions and context without masking unrelated I/O errors.
- Respect resource ownership: dispose streams opened by the library, and leave caller-owned streams open according to the API contract.
- Keep index inspection proportional to metadata and tensor descriptors. Do not read or copy gigabytes of weights just to report model information.
- For tensor conversion and future loading or decoding, make supported encodings, memory use, and resource lifetimes explicit. Do not silently reinterpret unsupported quantization formats.
- Keep quantization payload buffers bounded independently of tensor size. Preserve tensor shapes, typed metadata, and copied tensor bytes; recompute aligned offsets and update the file profile and quantization version. Distinguish generic tensor selection from architecture-specific quantization policies.
- Quantization must keep the input read-only and never overwrite an existing output. Validate a temporary sibling before publishing it, and clean it up on failure or handled cancellation. Reject unsupported encodings, endianness, split models and numerical values explicitly; do not silently enable requantization.
- Measure before optimizing. Distinguish index parsing, weight loading, decoding, and inference in timing results, and report the workload used.

## Use unsafe code only for large performance gains

- Write safe code by default. Arrays, `Span<T>`, `ReadOnlySpan<T>`, `BinaryPrimitives`, `MemoryMarshal.Cast`, and span-based `System.Runtime.Intrinsics` APIs cover most binary reading and numerical needs.
- Use unsafe code only when a measurement shows a large performance gain over the best safe version, for example in hot inference kernels or to access memory-mapped weights without copying. A small or unmeasured gain does not justify it. Unsafe code includes `NativePtr`, `fixed`, the `&&` address-of operator, `System.Runtime.CompilerServices.Unsafe`, and suppressing warning FS0009.
- Record the measurement that justifies unsafe code, with the workload and the safe and unsafe timings, next to the code or in the change description.
- Confine unsafe code to small private functions behind a safe API. Validate bounds, lengths, and alignment before entering it, and document the invariants it relies on, including the lifetime of pinned or mapped memory.
- Never apply unsafe code to file contents before they are validated.
- Keep the safe implementation as a reference, and test the unsafe version against it, including boundary cases.

## Keep scripts useful and reproducible

- Use `.fsx` files for exploration and inspection, reusing library code through `#load` or assembly references instead of duplicating the parser.
- Use `LM_STUDIO_MODELS_FOLDER` for the local LM Studio models root directory. Build model file paths from that root and relative path segments using `Path.Combine` or the `/>` helper. Keep display limits easy to adjust, and do not put machine-specific paths in tracked scripts, library defaults, or automated tests.
- Keep the variable's value in the local shell or IDE environment. F# Interactive must inherit it when starting, or it must be set explicitly in the session before loading the script.
- Launch profiles also inherit `LM_STUDIO_MODELS_FOLDER`; reference it as `%LM_STUDIO_MODELS_FOLDER%` in quoted `commandLineArgs` instead of committing a personal root directory.
- Summarize large metadata arrays, strings, and tensor lists in console output. Keep the complete parsed data available for further inspection.
- Do not commit downloaded weights or require a large model download for normal development checks.

## Validate relevant behavior

- For parser behavior changes, add focused tests for the affected contract or a regression. Prefer independently encoded binary fixtures to fixtures generated by the code under test.
- Cover relevant combinations of versions, byte order, metadata types, quantization block sizes, offsets, and malformed input. Include partial reads, truncation, overflow, and stream ownership when those contracts are affected.
- Keep tests deterministic and small. Test observable behavior rather than private implementation details or tautological copies of constants.
- For quantization, verify independently specified block bytes and reconstruction error, metadata and copied-payload preservation, chunk boundaries, failure cleanup, cancellation and overwrite protection. Do not require a local model or native llama.cpp installation.
- Use real local GGUF files as optional smoke checks, and distinguish those checks from automated fixture tests. Do not depend on the exploration script's model path in CI.
- Use the relevant commands from the repository root:

  ```sh
  dotnet build Marmotte.slnx
  dotnet test Marmotte.slnx
  dotnet fsi src/Marmotte.Gguf/GgufScript.fsx
  git diff --check
  ```

- The script command requires `LM_STUDIO_MODELS_FOLDER` and the local model selected by the script's relative path segments. See the [configuration instructions](README.md#inspect-a-local-gguf-file). Run it when changing model inspection or when a real-file check is useful, not as a mandatory test for every edit.
- Use `--no-restore` only after dependencies have been restored. When build servers cause problems, use `--disable-build-servers -m:1` instead of changing project behavior.
- Documentation-only changes need review and formatting checks, not a full test run. For code changes, run the relevant tests and build, then report what was actually verified and any remaining limitations.
