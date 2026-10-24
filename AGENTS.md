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
- `src/Marmotte.Gguf` owns the GGUF format types, binary reader, metadata, and tensor descriptors. Reading an index does not load, decode, or execute the model's weights.
- `src/Marmotte.Core` is intended for model and inference domain code. `src/Marmotte.Cli` is intended for command-line orchestration and presentation. Both are currently scaffolds; do not assume an inference pipeline already exists.
- Keep console output and application-specific decisions in the CLI or exploration scripts. Libraries should expose usable data and explicit contracts.
- `tests/Marmotte.Gguf.Tests` contains xUnit tests using small binary fixtures. Keep automated tests independent of local model installations.
- Keep dependencies directed toward reusable libraries. Do not make libraries depend on the CLI or scripts, or introduce architectural layers without a concrete need.

## Document contracts and intent

- Use F# `///` documentation for public APIs, important contracts, and non-obvious behavior.
- Explain why a decision exists, including constraints and tradeoffs, rather than restating the implementation.
- Document units, dimension order, supported versions, stream requirements, stream ownership, final stream position, and exception behavior where relevant.
- Make partial knowledge visible: an unknown tensor encoding has an unknown byte size, not a zero-byte payload; a file profile does not establish every tensor's encoding.
- Update the README, examples, and API documentation when a behavior or public contract changes.
- Verify binary format tags and layouts against the upstream specification or implementation. Record the relevant source near tables or in documentation instead of relying on remembered constants.

## Prefer readable, idiomatic F#

- Favor straightforward code over clever, dense expressions. Give functions and modules one cohesive responsibility.
- Prefer immutable bindings and pure transformations. Keep I/O and resource management at clear boundaries.
- Use records for related data and discriminated unions for alternatives with different meanings. Prefer exhaustive pattern matching to flags, casts, or loosely typed dictionaries.
- Use `option` for meaningful absence and `Result` for expected failures when designing a new API. Preserve established exception-based contracts where they are appropriate: the GGUF reader reports malformed files with `InvalidDataException` and lets ordinary I/O failures propagate.
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
- Preserve the distinction between offsets relative to the tensor data section and absolute positions in the source stream. Preserve dimensions in GGML order, with the contiguous dimension first.
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
- For future tensor loading or decoding, make supported encodings, memory use, and resource lifetimes explicit. Do not silently reinterpret unsupported quantization formats.
- Measure before optimizing. Distinguish index parsing, weight loading, decoding, and inference in timing results, and report the workload used.

## Keep scripts useful and reproducible

- Use `.fsx` files for exploration and inspection, reusing library code through `#load` or assembly references instead of duplicating the parser.
- Use `LM_STUDIO_MODELS_FOLDER` for the local LM Studio models root directory. Build model file paths from that root and relative path segments using `Path.Combine` or the `/>` helper. Keep display limits easy to adjust, and do not put machine-specific paths in tracked scripts, library defaults, or automated tests.
- Keep the variable's value in the local shell or IDE environment. F# Interactive must inherit it when starting, or it must be set explicitly in the session before loading the script.
- Summarize large metadata arrays, strings, and tensor lists in console output. Keep the complete parsed data available for further inspection.
- Do not commit downloaded weights or require a large model download for normal development checks.

## Validate relevant behavior

- For parser behavior changes, add focused tests for the affected contract or a regression. Prefer independently encoded binary fixtures to fixtures generated by the code under test.
- Cover relevant combinations of versions, byte order, metadata types, quantization block sizes, offsets, and malformed input. Include partial reads, truncation, overflow, and stream ownership when those contracts are affected.
- Keep tests deterministic and small. Test observable behavior rather than private implementation details or tautological copies of constants.
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
