# Metal kernel organization

Metal separates GPU code from host code, so each operation has two files with
the same name: `<Operation>.metal` for the compute function, and
`<Operation>.mm` for its exported host entry point.
[Add.metal](Add.metal) and [Add.mm](Add.mm) are the working example.

```text
native/
  CMakeLists.txt                Explicit source lists and compiler settings
  MetalContext.h                Device, queue, pipelines and allocation ownership
  MetalRuntime.mm               Initialization, allocation, transfers and synchronization
  kernels/
    README.md                   Shared development guide
    Add.metal                   F32, F16 and mixed GPU functions, in Metal Shading Language
    Add.mm                      Host entry point: validation, encoding and submission
```

These files appear in the F# project. CMake compiles the `.metal` sources into
one `.metallib` and embeds it in the `__TEXT,__metallib` section of the dynamic
library, so that deployment stays a single file. Split an operation further only
when its complexity justifies it.

## Compute function and host entry point

Name each compute function with an explicit format suffix, such as
`addValuesF32`, `addValuesF16` or `addValuesF32F16`.

| Function suffix | MSL storage type | Bytes per value |
| --- | --- | --- |
| `F32` | `float` | 4 |
| `F16` | `half` | 2 |

Bind buffers at fixed indices, with `device const` for inputs
and `device` for outputs, and read the element index from
`[[thread_position_in_grid]]`.

The exported `marmotte_metal_<operation>` function validates native allocations
and their recorded encodings, selects the compute function from them, gets the
pipeline with `findPipeline`, encodes one dispatch and calls
`commitCommandBuffer`. Public shape and ownership checks remain in
`MetalBackend.fs`. Keep allocation, transfers and synchronization in the runtime.

Document element counts, dimensions, aliasing and queue semantics beside the
entry point. Commit and return without waiting. Unsupported formats must fail
explicitly before any output write; never silently convert or reinterpret
storage to call an implemented variant.

Wrap the body of every entry point that reaches Metal in `@autoreleasepool`:
.NET threads have no pool of their own, so temporary objects would otherwise
accumulate for the life of the thread.

## Add an operation

1. Create `kernels/<Operation>.metal` and `kernels/<Operation>.mm`, using Add as
   the layout example. Choose the arguments from the Core contract; inputs and
   aliasing rules vary by operation.
2. List both files in `native/CMakeLists.txt`: the shader in
   `METAL_KERNEL_SOURCES`, the host file in `add_library`. Avoid globs that
   silently compile experimental sources.
3. Add a private P/Invoke binding in `MetalNative.fs` and implement the matching
   operation in `MetalBackend.fs`, validating all arguments before submission.
4. Test each supported precision against independent expected values: empty and
   scalar shapes when valid, threadgroup boundaries, special values, aliasing,
   rejection before writes and queued ordering. F16 expectations must account for
   rounding, and F32 ones for the [subnormal flush](#add-contract). Keep GPU tests
   opt-in.
5. Update the backend capability docs and examples. Build with
   `dotnet build Marmotte.slnx -p:MarmotteBuildMetal=true` and run the relevant
   [Metal tests](../../../../docs/metal.md#run-the-tests).

## Add contract

`Add` computes `destination[i] = left[i] + right[i]` over contiguous values. All
tensors must have identical shapes and encodings, F32 or F16, and belong to the
same backend instance. `count` counts elements, not bytes. The managed backend validates and
handles empty tensors before native dispatch.

`AddF32F16` computes the same sum with F32 left and destination tensors and an F16
right tensor. Add stays strict about encodings, so this widening is always
requested by name. The destination may be left, never right.

The native entry points check ownership, element counts and encodings from the
allocation registry. Each thread reads and writes only its own index, so the destination may
be either input or both, when encodings allow it. `dispatchThreads` runs exactly `count` threads, with
the pipeline's `threadExecutionWidth` as threadgroup size: a straightforward
starting point, not a benchmark-derived optimum. Execution errors surface at the
next wait.

Shaders are compiled with `-fno-fast-math`, because Metal enables fast math by
default. NaN, infinities and signed zero follow IEEE 754. NaN payloads after
arithmetic are unspecified. The GPU may flush F32 subnormal inputs and results to
zero, as an M4 Max does, while F16 subnormals are kept and F16 sums round to
nearest, ties to even; see
[floating-point behavior](../../../../docs/metal.md#floating-point-behavior).

[MetalTests.fs](../../../../tests/Marmotte.Backends.Tests/MetalTests.fs) covers
F32. [HalfTests.fs](../../../../tests/Marmotte.Backends.Tests/HalfTests.fs) shares
independent F16 and mixed fixtures between the CPU and Metal, including all 65,536
transfer bit patterns.
