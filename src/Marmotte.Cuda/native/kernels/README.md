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
    SwiGlu.cu                  F32/F16 GPU functions and host dispatch
    RmsNorm.cu                 F32/F16 GPU functions and host dispatch
    CopyRows.cu                F32/F16 device-to-device copy, no GPU function
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
without an F32 buffer. See [Half arithmetic and memory use](../../../../docs/kernels/add.md#half-add-and-memory-use).

[CudaTests.fs](../../../../tests/Marmotte.Backends.Tests/CudaTests.fs) covers
F32. [HalfTests.fs](../../../../tests/Marmotte.Backends.Tests/HalfTests.fs)
shares independent F16 fixtures and transfer/validation tests between CPU and
CUDA, including all 65,536 transfer bit patterns.

## SwiGlu contract and supported formats

`SwiGlu.cu` implements F32 and F16, both exposed through the managed backend.
The operation computes
`destination[i] = silu(gate[i]) × up[i]` with `silu(x) = x / (1 + e^-x)` over
contiguous values, using the device `expf` with no fast math. All tensors must
have identical shapes and encodings and belong to the same backend instance;
mixing F32 and F16 raises `ArgumentException` in `CudaBackend.fs`, as for Add,
and any other format returns `cudaErrorNotSupported` natively, before any output
write. `count` counts elements, not bytes. The managed backend validates and
handles empty tensors before native dispatch.

The F16 kernel widens both inputs exactly to F32 with `__half2float`, evaluates
the same F32 formula and rounds once with `__float2half_rn`, to nearest with ties
to even. Computing in half precision instead would round after each of the four
operations, and `e^-x` alone exceeds the largest finite half (65,504) once x is
below about -11.1, although the final product is still representable. Overflow
of the final product produces infinity; subnormal inputs and results are kept,
not flushed to zero.

Each thread reads its gate and up values and writes only its own index, so the
destination may be either input or both; do not add `__restrict__` promises that
would invalidate this contract. Launches use 256 threads per block and check the
final block's bounds. Because the CPU and the device approximate the exponential
differently, F32 results may differ by a few units in the last place: compare them
with a relative tolerance. Where `e^-x` overflows (x below about -88.7), the
F32 result is a signed zero, and `silu(-∞)` is NaN, as the formula gives; F16
inherits both behaviors through its F32 evaluation.

[CudaTests.fs](../../../../tests/Marmotte.Backends.Tests/CudaTests.fs) covers F32
against a double-precision reference. The CPU backend does not implement F16
SwiGLU, so [HalfTests.fs](../../../../tests/Marmotte.Backends.Tests/HalfTests.fs)
checks CUDA alone: exact bits for independently computed fixtures that lie far from
a rounding midpoint, and at most one unit in the last place from a double-precision
reference for all 65,536 gate bit patterns, since an F32 result within the
exponential's error of a midpoint may round to either neighbour.

## RmsNorm contract and supported formats

`RmsNorm.cu` implements F32 and F16, both exposed through the managed backend.
The operation computes
`destination[r, c] = input[r, c] / sqrt(mean(input[r, *]²) + epsilon) × weight[c]`
per row of the last dimension. Input and destination have the same shape of rank 1
or more, and weight is a vector of the last dimension's length; all belong to the
same backend instance and share one encoding — mixing F32 and F16 raises
`ArgumentException` in `CudaBackend.fs`, as for Add and SwiGlu, and any other
format returns `cudaErrorNotSupported` natively, before any output write. `rows`
and `columns` count elements, not bytes. The managed backend validates shapes and
epsilon and handles empty tensors before native dispatch.

One block of 256 threads handles one row. Each thread accumulates its strided
share of the sum of squares in double precision, and a shared-memory block
reduction adds the partials in double: rows of thousands of values keep F32
accuracy, and squares beyond the F32 range, such as that of 1e20, do not overflow.
The scale `1 / sqrt(mean + epsilon)` is computed in double and rounded once to F32,
then applied as `value * scale * weight` with two F32 roundings, matching the CPU
reference. The reduction completes before any write, so destination may be input,
and each column is read from weight by the same thread that writes its destination.
The mean is a reduction that implementations sum in different orders, so compare
results with a relative tolerance. NaN or infinite inputs make their row's results
NaN or zero, as the formula gives.

The F16 kernel widens both inputs exactly to F32 with `__half2float`, runs the same
double-precision reduction and F32 scale, and rounds the final product once with
`__float2half_rn`, to nearest with ties to even. Half intermediates would round
after every operation, and the largest half (65504) already loses accuracy when
squared in F32. Overflow of the final product produces infinity; subnormal inputs
and results are kept, not flushed to zero. The CPU backend implements RMS
normalization in F32 only and rejects F16 with `NotSupportedException`.

[CudaTests.fs](../../../../tests/Marmotte.Backends.Tests/CudaTests.fs) covers F32
against a double-precision reference. [HalfTests.fs](../../../../tests/Marmotte.Backends.Tests/HalfTests.fs)
checks CUDA F16: exact bits for independently computed fixtures that lie far from
a rounding midpoint, and at most one unit in the last place from a double-precision
reference across every binary16 input pattern.

## CopyRows contract and supported formats

`CopyRows.cu` copies F32 and F16 rows, both exposed through the managed backend:
`destination[r, c] = source[firstRow + r, c]` for a rank-2 `[sourceRows; columns]`
source and a `[rows; columns]` destination. It is the one operation without a
GPU function. Contiguous row-major rows form a single byte range, so the entry
point issues one `cudaMemcpyAsync` with `cudaMemcpyDeviceToDevice` on the
context's stream. The copy engine moves the bytes in stream order without a
launch configuration; a hand-written kernel would only repeat that work.
The file still follows the one-operation layout: its exported entry point validates
the registry before any copy, and the runtime keeps host transfers.

Values are copied bit for bit, including F16 NaN payloads; there is no arithmetic,
rounding or conversion. Both tensors must belong to the same backend instance and
share one encoding; mixing F32 and F16 raises `ArgumentException` in
`CudaBackend.fs`, and any other format returns `cudaErrorNotSupported` natively.
`sourceRows`, `firstRow`, `rows` and `columns` count rows and elements, not bytes.
The managed backend validates rank, column counts and the range
`0 <= firstRow <= sourceRows - rows` before dispatch, and handles empty
destinations, including `firstRow = sourceRows`, without native calls.

The native entry point checks again that the source and destination are execution
allocations registered with exactly `sourceRows × columns` and `rows × columns`
elements. It computes those products and the byte offset in 64-bit arithmetic,
so stored weights, foreign pointers and inconsistent counts are rejected before a
copy. Distinct allocations never overlap. The contract's only aliasing case is
the identity copy, the same tensor with `firstRow = 0` and every row; it returns
without a copy, because overlapping device copies are undefined.

[CudaTests.fs](../../../../tests/Marmotte.Backends.Tests/CudaTests.fs) compares every
start row and length with the CPU, checks F16 bits including NaN payloads, the
identity copy, stream order between Add operations queued before and after the copy,
and rejection before writes. [CudaWeightTests.fs](../../../../tests/Marmotte.Backends.Tests/CudaWeightTests.fs)
calls the C ABI directly with inconsistent counts, ranges, encodings, stored weights and
another context.
