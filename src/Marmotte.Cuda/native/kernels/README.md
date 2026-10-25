# CUDA kernel organization

Keep one `.cu` file per operation. Its typed GPU functions and shared exported
host entry point live together. [Add.cu](Add.cu) is the working example.

```text
native/
  CMakeLists.txt                Explicit source list and compiler settings
  CudaContext.h                 Device, stream and allocation ownership
  CudaRuntime.cu                Allocation, transfers and synchronization
  kernels/
    README.md                  Shared development guide
    Add.cu                     F32/F16 GPU functions and host dispatch
```

These files appear in the F# project. CMake compiles the `.cu` sources.
Split an operation further only when its complexity justifies it; a new
operation does not need its own header, CMake file or directory by default.

## Typed functions and host dispatch

| Function suffix | CUDA storage type | Bytes per value |
| --- | --- | --- |
| `F32` | `float` | 4 |
| `F16` | `half`, provided by `cuda_fp16.h` | 2 |

**F16 and Half mean the same format.** `half2` can process pairs of F16 values;
it is not another tensor format. See the type and arithmetic references in
[docs/cuda.md](../../../../docs/cuda.md#kernel-formats-and-file-layout).

Keep GPU functions private in an anonymous namespace, with explicit F32/F16
names and typed pointers. The exported `marmotte_cuda_<operation>` function
validates native allocations and selects the device, then dispatches from the
recorded encoding. Its `void*` parameters allow one C ABI entry point to accept
either format; casts occur only after format validation. Public shape and
ownership checks remain in `CudaBackend.fs`.

Document element counts, dimensions, aliasing and stream semantics beside the
entry point. Launch on the supplied context's stream and return the launch
status without waiting. Keep allocation, transfers and synchronization in the
runtime. Unsupported formats must fail explicitly before any output write;
never silently convert or reinterpret storage to call an implemented variant.

## Add an operation

1. Create `kernels/<Operation>.cu` using Add as the layout example. Choose its
   arguments from the Core contract; inputs and aliasing rules vary by operation.
2. Implement its typed F32 and F16 GPU functions and one shared C entry point.
   If only one format is implemented, reject the other explicitly and document it.
3. List the file in `native/CMakeLists.txt`. Avoid globs that silently compile
   experimental sources.
4. Add a private P/Invoke binding in `CudaNative.fs` and implement the matching
   operation in `CudaBackend.fs`, validating all arguments before submission.
5. Test each supported precision against independent expected values: empty
   and scalar shapes when valid, block boundaries, special values, aliasing,
   rejection before writes and queued ordering. F16 expectations must account
   for rounding; copying F32 expectations is insufficient. Keep GPU tests opt-in.
6. Update the backend capability docs and examples. Build with
   `dotnet build Marmotte.slnx -p:MarmotteBuildCuda=true` and run the relevant
   [CUDA tests](../../../../docs/cuda.md#run-the-tests).

Storage and arithmetic precision are separate decisions. A future F16
reduction may accumulate in scalar F32, while its tensor stays in F16.
Document and test rounding and accumulation without creating an F32 shadow
buffer. New formats must extend allocation, transfers, managed metadata and
tests before becoming callable.

## Add contract and supported formats

Both F32 and F16 are implemented in `Add.cu` and exposed through the managed
backend. The operation computes `destination[i] = left[i] + right[i]` over
contiguous values. All tensors must have identical shapes and encodings and
belong to the same backend instance. `count` counts elements, not bytes. The
managed backend validates and handles empty tensors before native dispatch.

The native entry point checks the registry's ownership, element counts and
encodings before selecting `addValuesF32` or `addValuesF16`. Each thread reads
and writes only its own index. The destination may be either input or both;
do not add `__restrict__` promises that would invalidate this contract. Launches
use 256 threads per block, check the final block's bounds and enable no fast
math. Execution errors may surface at later synchronization.

F32 performs ordinary single-precision addition. F16 uses `__hadd` with
two-byte inputs and output, rounding to nearest with ties to even. Overflow
produces infinity; subnormals and signed zero are supported. NaN payloads after
arithmetic are unspecified. The runtime allocates two bytes per F16 element
and transfers raw bits through `uint16[]` staging, preserving even NaN payloads
without an F32 buffer. See [Half arithmetic and memory use](../../../../docs/kernels.md#half-add-and-memory-use).

[CudaTests.fs](../../../../tests/Marmotte.Backends.Tests/CudaTests.fs) covers
F32. [HalfTests.fs](../../../../tests/Marmotte.Backends.Tests/HalfTests.fs)
shares independent F16 fixtures and transfer/validation tests between CPU and
CUDA, including all 65,536 transfer bit patterns.
