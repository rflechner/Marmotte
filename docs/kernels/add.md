# Add and Buffer Reuse

[Back to execution tensors and kernels](../kernels.md).

Learn how matching tensor positions are added and when their storage can be reused.
Prerequisites: arrays and tensor shapes; see the [notation guide](../kernels.md#reading-the-formulas).
Run `dotnet fsi examples/Add.fsx` from the repository root with .NET 10 to try
F32 and F16 addition without a model file.

![Add: at each of three positions, add the two input values; [1, -2, 3] plus [4, 5, -1] gives [5, 3, 2]](add.png)

[Open the Add illustration](add.png). Its backend footer is historical:
CUDA and Metal now implement Add for F32 and F16, as described below.

Read `i` as the position being processed. At position `1`, read `-2` from
`left` and `5` from `right`, then write `3` to the destination. Every position
produces its own output; the output keeps the input shape. **In-place** means
reusing an input's storage for that destination once its values have been read.

```fsharp
open System
open Marmotte.Core
open Marmotte.Cpu

let addExample () =
    use backend = new CpuBackend() :> ITensorBackend
    let shape = TensorShape.Create [| 2; 3 |]
    use left = backend.Upload(shape, ReadOnlyMemory<single>([| 1.0f; 2.0f; 3.0f; 4.0f; 5.0f; 6.0f |]))
    use right = backend.Upload(shape, ReadOnlyMemory<single>([| 10.0f; 20.0f; 30.0f; 40.0f; 50.0f; 60.0f |]))
    use destination = backend.Allocate(shape)

    backend.Add(left, right, destination)

    let result = Array.zeroCreate<single> shape.ElementCount
    backend.Download(destination, result.AsMemory())
    result // [| 11.0f; 22.0f; 33.0f; 44.0f; 55.0f; 66.0f |]
```

`Add(left, right, destination)` writes `destination[i] = left[i] + right[i]`.
All three tensors must have exactly the same dimensions and encoding, not just
the same number of elements. There is no broadcasting, type promotion or strided view.
Empty tensors are valid no-ops; scalars add to scalars. NaN and infinities follow
ordinary F32 addition rather than being rejected as malformed model data.

For F32, the CPU backend processes `Vector<single>.Count` values at a time: four with
ARM64 NEON, eight with AVX2. It reinterprets its arrays as vectors with
`MemoryMarshal.Cast`, without unsafe code, and adds the remaining values one by
one. Each lane performs the same single-precision addition as a scalar loop, so
the results are identical. Kernels that reduce values, such as dot products,
will not have that property: vectorizing them changes the summation order, and
their tests will need tolerances.

The destination is explicit so a caller can reuse its buffer. Exact aliasing is
supported: `backend.Add(hidden, residual, hidden)` updates the hidden state in
place, and `backend.Add(hidden, hidden, hidden)` doubles it. Otherwise inputs are
unchanged. All arguments are validated before any destination write.

Every tensor must come from the **same backend instance** executing the kernel.
`BackendKind.Cpu` alone is insufficient: even another `CpuBackend` instance is a
different ownership domain. This leaves room for distinct devices and contexts
without treating their allocations as interchangeable. An explicit download and
upload is currently the only way to move values between instances.

## Half Add and memory use

F16 is IEEE binary16, represented by .NET's
[`System.Half`](https://learn.microsoft.com/dotnet/api/system.half) and CUDA/MSL
`half`. CPU, CUDA and Metal implement Add for this format, including in-place Add.
The same code works with each backend:

```fsharp
let addHalf (backend: ITensorBackend) =
    let shape = TensorShape.Create [| 3 |]
    let leftValues = [| 1.0f; 2.0f; 3.0f |] |> Array.map Half.op_Explicit
    let rightValues = [| 10.0f; 20.0f; 30.0f |] |> Array.map Half.op_Explicit
    use left = backend.Upload(shape, ReadOnlyMemory<Half>(leftValues))
    use right = backend.Upload(shape, ReadOnlyMemory<Half>(rightValues))
    use destination = backend.Allocate(shape, TensorEncoding.F16)
    backend.Add(left, right, destination)
    let result = Array.zeroCreate<Half> shape.ElementCount
    backend.Download(destination, result.AsMemory())
    result // Half values: 11, 22, 33
```

The conversion of literals above is explicit application code. Internally, an
F16 tensor has **2 × element count bytes of payload**, compared with 4 × count
for F32; allocator and object overhead are additional. The CPU owns `Half[]`.
CUDA allocates `count * sizeof(half)` bytes and copies the raw bits through a
temporary `uint16[]` staging array, also two bytes per element. Metal owns a shared
`MTLBuffer` with two bytes per element. None creates an F32 shadow tensor or F32 transfer buffer. Upload and Download preserve
all bits, including signed zero and NaN payloads.

Storage precision and arithmetic implementation are distinct. The safe CPU
loop uses `Half` addition; the runtime can promote individual operands to scalar
F32 and round back to Half. This does not expand the tensor into an F32 array.
CUDA uses [`__hadd`](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__ARITHMETIC.html)
in a kernel with `half*` inputs and output. Metal adds MSL `half` values directly.
These paths round to nearest F16,
ties to even, retain subnormal results, and produce infinity on overflow.
Arithmetic NaN payloads are unspecified. This smaller format has less range
and precision; it is not an automatic performance improvement, and no speedup
is claimed for the scalar CPU implementation.

`AddF32F16(left, right, destination)` adds an F16 tensor into an F32 one, such as
F16 projection output into an F32 residual stream, on CPU and Metal. Left and
destination are F32 and right is F16; destination may be left. `Add` rejects mixed
encodings, so this widening is always requested by name. Every F16 value converts
exactly to F32, so the only rounding is that of the F32 sum. CUDA leaves this
mixed operation unimplemented.

`HalfTests.fs` checks hand-specified binary16 results at rounding boundaries,
overflow, normal/subnormal transitions, signed zero, infinities and NaNs, as
well as transfer of all 65,536 bit patterns. It exercises the same cases on CPU
and, when explicitly enabled, CUDA and Metal. On CPU, kernels other than Add,
`AddF32F16` and `RmsNormF32ToF16` require F32 and reject F16 with
`NotSupportedException` before writing; mixed-format Add and incorrect Download
overloads raise `ArgumentException`, including for empties.
