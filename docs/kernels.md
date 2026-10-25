# Execution Tensors and the Add Kernel

`Marmotte.Core` defines the compute contracts. `Marmotte.Cpu`, `Marmotte.Cuda`
and `Marmotte.Metal` each reference Core; Core references none of the backends.
The first implemented operation is element-wise F32 addition.

## From GGUF to Add

![GGUF index reading, requested payload loading, weight placement, planned model computation, F32 Add and CPU/CUDA/Metal backends](gguf-to-add-kernel.png)

[Open the full-size diagram](gguf-to-add-kernel.png).

The diagram follows the implementation as of 2026-10-25. Solid teal paths are
implemented, dashed amber paths are planned, and gray panels identify GPU
placeholders. Opening a source reads the index; `WeightMatrix.Read` requests a
payload through that source. CPU `UploadWeights` then retains the encoded
matrix without another payload copy.

The missing bridge is model computation: future embedding and matrix-product
kernels will consume stored weights and produce F32 activations. `IAddKernel`
adds those activations, not encoded weights. Today, `examples/Add.fsx` supplies
F32 host values directly through `Upload`. The output can feed another `Add`
without a download; `Download` is the explicit route back to host values.

The image was generated with the built-in `image_gen` tool and visually reviewed.
Its [generation and correction prompts](gguf-to-add-kernel.prompt.txt) are saved
alongside it. The contracts below remain the authoritative description.

## Stored Weights and Execution Tensors

`ITensorSource` and `WeightMatrix` provide read-only access to encoded model
weights. Their payloads can contain quantization blocks, and they cannot be
passed directly to `Add`. `backend.UploadWeights(matrix)` places a matrix on a
backend as an `IWeightTensor`, keeping its encoding, rows and columns: the
weights are never decoded as a whole. The CPU backend retains the matrix payload
without copying it; a GPU backend is expected to copy it into device memory.
Because a backend may retain the payload, keep the tensor source that produced
the matrix alive while the weight tensor is in use. No kernel consumes weight
tensors yet; a matrix-vector product would take an `IWeightTensor` and F32
`ITensor` values.

`ITensor` represents an owned, writable, contiguous **F32 execution tensor**.
Its `TensorShape` uses row-major dimensions: `[2; 3]` means two rows of three
values. Shapes copy their dimension arrays and expose a read-only collection.
An empty shape is a scalar with one value; any zero dimension makes an empty
tensor. Negative dimensions and element counts beyond `Array.MaxLength` are
rejected before allocating storage. Allocation still depends on available memory.

The tensor interface exposes shape, encoding and backend family, but no array,
span or native pointer. Each implementation owns its storage. `Upload` accepts
host `ReadOnlyMemory<single>` and copies it into new storage; `Download` copies
into caller-provided `Memory<single>`. Both require an exact element count.
This makes transfers explicit without requiring a GPU allocation to masquerade
as a managed array. Model weights use `UploadWeights` instead, and nothing
decodes them automatically.

## One Contract, Three Implementations

| Concern | CPU today | CUDA placeholder | Metal placeholder |
| --- | --- | --- | --- |
| Factory | `CpuBackend()` | `CudaBackend()` | `MetalBackend()` |
| Owned tensor storage | Private `single array` | Future device allocation | Future Metal buffer |
| `Allocate` | Zero-filled array | `NotImplementedException` | `NotImplementedException` |
| `Upload` / `Download` | Explicit host copies | `NotImplementedException` | `NotImplementedException` |
| `UploadWeights` | Retains the payload without copying | `NotImplementedException` | `NotImplementedException` |
| `Synchronize` | Nothing to wait for | `NotImplementedException` | `NotImplementedException` |
| `IAddKernel.Add` | Safe `Vector<T>` loop with a scalar tail | `NotImplementedException` | `NotImplementedException` |
| SDK/driver required today | None | None | None |

`ITensorBackend` inherits `IAddKernel` so orchestration can call `backend.Add`.
A consumer needing only addition can accept `IAddKernel` directly. No generic
operation registry, graph scheduler or native interop is needed for this example.
GPU factories and disposal work on any host; all other operations fail explicitly,
without a CPU fallback. Concrete GPU tensor types will be introduced with their
storage implementation, rather than exposing unusable device handles now.

## Add and Buffer Reuse

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
All three tensors must have exactly the same dimensions, not just the same
number of elements. There is no broadcasting, type promotion or strided view.
Empty tensors are valid no-ops; scalars add to scalars. NaN and infinities follow
ordinary F32 addition rather than being rejected as malformed model data.

The CPU backend processes `Vector<single>.Count` values at a time: four with
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

## Ordering, Completion and Lifetime

Operations submitted to one backend instance take effect in submission order: a
kernel sees the results of earlier kernels and transfers. A kernel may return
before its result is computed, so a GPU backend can queue a whole forward pass
without waiting for the device after each operation. `Upload` and
`UploadWeights` copy or retain their input before returning, so a host buffer
can be reused at once. `Download` waits for every earlier operation before
copying, and `Synchronize` waits without copying. A device failure in a queued
operation surfaces on a later call to the same backend, at the latest at the
next `Download` or `Synchronize`. The CPU backend completes every operation
before returning, which satisfies these rules; its `Synchronize` has nothing to
wait for.

Callers own the tensors a backend returns and dispose them when they no longer
need them. Disposing a tensor is ordered after the operations already submitted
that use it, so a GPU backend must defer releasing device memory until they
finish. Disposing the backend waits for submitted operations, then releases
every tensor it still owns: device memory is never released after its context,
and a forgotten tensor does not outlive its backend. Those tensors become
disposed, and disposing them again is a no-op. All disposals are idempotent,
and metadata stays readable afterwards. Operations on disposed storage or a
disposed backend raise `ObjectDisposedException`; invalid arguments raise
`ArgumentException` (or `ArgumentNullException`). Concurrent calls on one
backend, including disposal, are outside this initial contract.

## Run the Example

From the repository root, with no model files or GPU installation:

```sh
dotnet fsi examples/Add.fsx
```

The script loads the library sources, runs the same function with each backend,
prints the CPU result, and reports the GPU placeholders' exceptions at the first
upload. Tests also call GPU allocation, download and `Add` directly to verify
that all unfinished operations fail explicitly.
