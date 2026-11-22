# CUDA: from the first build to an Add on the GPU

Marmotte's CUDA backend implements contiguous F32 and F16 tensors, zero-initialized
allocation, uploads, downloads, stored weight uploads, and element-wise Add and
SwiGLU in both formats. It uses
native CUDA C++ through P/Invoke, with no additional NuGet dependency. Other
kernels and the key-value cache still throw
`NotImplementedException`.

This guide assumes familiarity with building the .NET solution, but no CUDA
experience. Commands run from the repository root. The native build is optional:
`dotnet build Marmotte.slnx` and the default tests work without CUDA installed.

## What the different tools do

| Component | Purpose |
| --- | --- |
| NVIDIA GPU | Executes the parallel arithmetic and holds tensors in device memory (VRAM). |
| NVIDIA driver | Lets the operating system and CUDA runtime communicate with the GPU. |
| CUDA Toolkit | Development tools, headers and libraries; includes the `nvcc` compiler. |
| Host C++ compiler | Compiles the CPU portion of a CUDA source file: MSVC on Windows, a supported GCC toolchain on Linux. |
| CMake | Generates a native build for the selected toolchain, then invokes it. |
| .NET SDK | Compiles the F# backend and applications. With `MarmotteBuildCuda=true`, MSBuild also invokes CMake to compile the `.cu` files with `nvcc`. |

A `.cu` file can contain CPU functions and GPU functions. `nvcc` compiles both
parts, delegating CPU compilation to the host compiler. Our result is a shared
library: `marmotte_cuda.dll` on Windows or `libmarmotte_cuda.so` on Linux. The
F# assembly remains a separate `Marmotte.Cuda.dll` on both platforms.

`nvidia-smi` reports the driver and GPU. Its displayed CUDA version describes
driver support, not the Toolkit installed for compilation. Use `nvcc --version`
to inspect the compiler. See NVIDIA's [installation guide](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-installation-guide-microsoft-windows/index.html)
and [driver compatibility explanation](https://docs.nvidia.com/deploy/cuda-compatibility/).

## Install and check prerequisites

Use a 64-bit Windows or Linux process and an NVIDIA GPU supported by your
Toolkit and driver. The managed solution targets .NET 10. The CMake project
requires CMake 3.24 or later and CUDA C++17 support.

On Windows, install these components:

1. An NVIDIA driver compatible with the chosen Toolkit.
2. Visual Studio or Build Tools with **Desktop development with C++** and a
   Windows SDK. The .NET workload alone does not provide a native compiler.
3. The CUDA Toolkit, including its Visual Studio build integration.
4. CMake and the .NET 10 SDK.

Choose the host compiler from the compatibility table for your Toolkit version;
installing the newest Visual Studio does not automatically make it compatible.
The commands below select Visual Studio 2022 explicitly. They were exercised
with CUDA 12.8, MSVC 19.42, CMake 4.1.2 and an RTX 4070 Ti SUPER (compute
capability 8.9), with driver 610.88 on Windows. This records a working setup,
not a requirement to use those exact versions.

Check from PowerShell:

```powershell
dotnet --version
cmake --version
nvcc --version
nvidia-smi
```

The Visual Studio CMake generator finds MSVC without requiring `cl.exe` on
your ordinary shell's PATH. Ninja instead needs a developer shell with the
compiler environment configured. For Linux, follow NVIDIA's
[Linux installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
for the supported distribution, driver, Toolkit and host compiler combination.

## Build on Windows

Enable native compilation directly in the managed build:

```powershell
dotnet build Marmotte.slnx -p:MarmotteBuildCuda=true
dotnet fsi examples/CudaAdd.fsx
```

Expected example output:

```text
CUDA Add, then in-place Add: [|12.0f; 24.0f; 36.0f; 48.0f; 60.0f; 72.0f|]
CUDA Half Add, then in-place Add: [|12; 24; 36; 48; 60; 72|]
```

The `BuildCudaNative` target in
[`Marmotte.Cuda.fsproj`](../src/Marmotte.Cuda/Marmotte.Cuda.fsproj) configures and
builds CMake before the F# project is compiled. It also runs before collecting
files to copy to consumers. The resulting library is copied beside the managed
assembly and propagated through project references to applications/tests and
publish output. A failed native build stops the managed build.

The target runs CMake on each real build so edits to `.cu`, `.h` and CMake files
are considered even when the F# sources are unchanged. CMake's generated build
system handles incremental compilation. Visual Studio's fast up-to-date shortcut
is disabled when CUDA compilation is enabled; design-time builds used by the
editor never invoke CMake. These hooks use MSBuild's
[`BeforeTargets`](https://learn.microsoft.com/en-us/visualstudio/msbuild/how-to-extend-the-visual-studio-build-process)
and [`Exec`](https://learn.microsoft.com/en-us/visualstudio/msbuild/exec-task).

| MSBuild property | Default and purpose |
| --- | --- |
| `MarmotteBuildCuda` | `false`; set to `true` or `1` to configure/build CUDA before the managed project. Both environment variables and `-p:` properties accept these values. |
| `MarmotteCudaArchitectures` | `native`, targeting GPU(s) on the build machine. Set a numeric target to build without a GPU. |
| `MarmotteCudaGenerator` | `Visual Studio 17 2022` on Windows, CMake's default on Linux. Visual Studio generators use `-A x64`. |
| `MarmotteCudaBuildDirectory` | `artifacts/cuda/msbuild/<Configuration>` under the repository; an override should be an absolute directory path. |
| `MarmotteCudaNativePath` | An existing native library to deploy instead of building it. Do not combine with `MarmotteBuildCuda=true`. |

For example, target compute capability 8.9 explicitly:

```powershell
dotnet build Marmotte.slnx -p:MarmotteBuildCuda=true -p:MarmotteCudaArchitectures=89
```

`89` means compute capability 8.9, not CUDA version 8.9. The Toolkit must
recognize the requested target. Unsuffixed numeric architectures request both
device machine code and PTX, the intermediate GPU code that a compatible driver
can compile when loading. See the [CMake architecture options](https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html).
For a list in an MSBuild command, escape its semicolons, for example
`"-p:MarmotteCudaArchitectures=86%3B89"`.

The native configuration follows the managed configuration: Debug by default,
Release with `-c Release`. Separate build directories prevent those configurations
from sharing a native output file. The example references managed **Debug**
assemblies; adjust its `#r` paths when using Release. Pass `MarmotteBuildCuda=true`
to `dotnet publish` as well when building an application for deployment.

Keep the property on subsequent builds: a build without it can remove a previously
copied native library as an obsolete output. To enable CUDA for all builds from
one PowerShell session, set `$env:MarmotteBuildCuda = 'true'`; MSBuild reads this
environment variable too. Remove it with `Remove-Item Env:MarmotteBuildCuda` to
return to managed-only builds.

## Build and run CUDA tests in Rider

Two settings control different stages:

| Setting | Read by | Purpose |
| --- | --- | --- |
| `MarmotteBuildCuda=true` or `1` | MSBuild, during the build | Compiles the native library and copies it beside the test assemblies. |
| `MARMOTTE_TEST_CUDA=1` | The test process | Enables the optional GPU tests; it does not compile or deploy the native library. |

If `MarmotteBuildCuda` is a Windows user environment variable, restart Rider
after setting it so its build processes inherit the new value. Restart its
launcher too if Rider is launched from a process that still has the old
environment. Setting a variable in Rider's terminal only affects that terminal
and its child processes. A variable configured under **Unit Testing > Test
Runner > Environment variables** is passed to the test process, not the build.
Use that test setting for `MARMOTTE_TEST_CUDA=1`; see Rider's
[test runner environment settings](https://www.jetbrains.com/help/rider/Reference__Options__Tools__Unit_Testing__Test_Runner.html).

In **Settings > Build, Execution, Deployment > Toolset and Build**, disable
**Use ReSharper Build** for this solution's native CUDA workflow, then rebuild
the solution. ReSharper Build can skip projects it considers unchanged,
including their custom CMake steps. Delegating the build to MSBuild ensures our
target runs and lets CMake decide which native sources need recompilation.
JetBrains documents this limitation for [custom build steps](https://www.jetbrains.com/help/rider/Build_Process.html).

For Debug on Windows, the build must produce:

```text
tests/Marmotte.Backends.Tests/bin/Debug/net10.0/marmotte_cuda.dll
```

It belongs beside `Marmotte.Cuda.dll` and the test assembly, not only in
`artifacts/cuda/`. Release uses the corresponding Release folder. An ordinary
build without CUDA enabled can remove a previously copied DLL, so keep the
build setting enabled when Rider rebuilds before running the tests.

To check the build and deployment independently of Rider, run from PowerShell:

```powershell
$env:MarmotteBuildCuda = '1'
dotnet build tests/Marmotte.Backends.Tests/Marmotte.Backends.Tests.fsproj
Test-Path tests/Marmotte.Backends.Tests/bin/Debug/net10.0/marmotte_cuda.dll
$env:MARMOTTE_TEST_CUDA = '1'
dotnet test tests/Marmotte.Backends.Tests/Marmotte.Backends.Tests.fsproj --no-build --filter Category=CUDA
```

`Test-Path` must return `True`. If these tests pass but Rider still fails, check
the build output and environment inherited by Rider, and rebuild before running
the tests again. The command-line checks validate the library and tests; they
do not validate Rider's current settings.

## Build on Linux

The same source and C ABI support a Linux shared library. This path has not
been exercised by the Windows validation described below.

```sh
dotnet build Marmotte.slnx -p:MarmotteBuildCuda=true
dotnet fsi examples/CudaAdd.fsx
```

MSBuild passes the configuration as `--config`, and also as `CMAKE_BUILD_TYPE`
for generators other than Visual Studio and Ninja Multi-Config.
If `nvcc` is not on PATH, set the `CUDACXX` environment variable to its absolute
path before the first configuration. Use a new `MarmotteCudaBuildDirectory` when
changing compilers or generators; CMake caches the selected toolchain.

## Build native code separately or use a prebuilt library

The separate CMake workflow remains available, for example to build an optimized
native library while debugging managed code:

```powershell
cmake -S src/Marmotte.Cuda/native -B artifacts/cuda -G "Visual Studio 17 2022" -A x64 -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build artifacts/cuda --config Release
$cudaLibrary = (Resolve-Path artifacts/cuda/native/marmotte_cuda.dll).Path
dotnet build Marmotte.slnx "-p:MarmotteCudaNativePath=$cudaLibrary"
```

`-S` identifies sources; `-B` keeps generated files in a separate directory;
`-G` selects the build system. `--config Release` selects the optimized native
configuration. For Linux, omit the Visual Studio generator and `-A x64`, add
`-DCMAKE_BUILD_TYPE=Release`, and use `libmarmotte_cuda.so` as the output filename.

`MarmotteCudaNativePath` takes an **absolute file path**, not a directory. In this
mode the project only copies the supplied binary; it does not invoke CMake.
Rebuild native code separately after editing it, then run the managed build
with the property again. Use this property on `dotnet publish` to package a
prebuilt native library. No files need to be copied to a system directory.

The native target links the CUDA runtime statically, using CMake's
[`CUDA_RUNTIME_LIBRARY`](https://cmake.org/cmake/help/latest/prop_tgt/CUDA_RUNTIME_LIBRARY.html)
property. The NVIDIA driver is still required at execution time. On Windows,
the C/C++ runtime is also linked statically. Linux still requires compatible
system C/C++ libraries. We do not package a driver or prebuilt native binaries.

## Follow one call through the code

| File | Responsibility |
| --- | --- |
| [`CudaBackend.fs`](../src/Marmotte.Cuda/CudaBackend.fs) | Tensor shapes, instance ownership, disposal and the public Core interfaces. |
| [`CudaNative.fs`](../src/Marmotte.Cuda/CudaNative.fs) | P/Invoke declarations, native error translation and the context `SafeHandle`. |
| [`native/CudaContext.h`](../src/Marmotte.Cuda/native/CudaContext.h) | Private shared state: selected device, stream and owned allocations with their element counts, encodings and byte sizes. |
| [`native/CudaRuntime.cu`](../src/Marmotte.Cuda/native/CudaRuntime.cu) | Device initialization, allocation, copies, synchronization and cleanup. |
| [`native/kernels/Add.cu`](../src/Marmotte.Cuda/native/kernels/Add.cu) | F32/F16 GPU functions and the shared CPU entry point, `marmotte_cuda_add`: native validation, format dispatch and launch configuration. |
| [`native/kernels/SwiGlu.cu`](../src/Marmotte.Cuda/native/kernels/SwiGlu.cu) | F32/F16 GPU functions and the shared CPU entry point, `marmotte_cuda_swiglu`: native validation, format dispatch and launch configuration. |
| [`native/CMakeLists.txt`](../src/Marmotte.Cuda/native/CMakeLists.txt) | Native sources, compiler settings and output location. |

```mermaid
sequenceDiagram
    participant App as F# application
    participant Backend as CudaBackend
    participant Native as Native library
    participant GPU as GPU stream
    App->>Backend: Upload(shape, host values)
    Backend->>Native: Allocate and upload
    Native->>GPU: Allocate, zero, copy host to device
    Native->>GPU: Wait for stream
    Backend-->>App: Owned ITensor
    App->>Backend: Add(left, right, destination)
    Backend->>Backend: Validate shapes, owners and lifetimes
    Backend->>Native: marmotte_cuda_add
    Native->>GPU: Queue addValuesF32 or addValuesF16
    Backend-->>App: Return after submission
    App->>Backend: Download(destination, host buffer)
    Backend->>Native: Download
    Native->>GPU: Copy device to host after Add, then wait
    Backend-->>App: Completed host buffer
```

For `[2; 3]`, storage is six consecutive values in row-major order: 24 bytes in
F32 or 12 bytes in F16. Shape
validation stays in F#; the native addition only needs the element count.
`extern "C"` gives exported functions stable names without C++ name mangling.
P/Invoke calls those CPU functions; the CPU function launches GPU code.

`Add.cu` contains `marmotte_cuda_add`, which runs on the CPU and selects
`addValuesF32` or `addValuesF16` from the allocation registry. It rejects mixed
formats before launching anything. Both typed functions, marked `__global__`,
run on the GPU and live in the same file. The shared host entry point launches
the selected function directly. The launch uses
`<<<blocks, threads, 0, stream>>>`: a grid of blocks, 256 threads per block,
zero bytes of dynamic shared memory, and this backend's stream. A thread's
element index is `blockIdx.x * blockDim.x + threadIdx.x`. For 257 values, two
blocks launch 512 threads; the bounds check lets only indices 0 through 256
write. No threads communicate, and each output reads only the same index of
each input. That also makes exact in-place Add valid. NVIDIA's
[programming guide](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-c-programming-guide/index.html#programming-model)
explains the underlying thread hierarchy and launch syntax.

The block size is a straightforward starting point, not a benchmark-derived
optimum. No fast-math flags are enabled. Tests cover NaN, infinities, signed zero
and subnormal values as well as ordinary additions; NaN payload bits are not a
portable contract.

## Memory, ordering and lifetime

`new CudaBackend(deviceIndex = 0)` selects a zero-based CUDA device ordinal.
Construction does not load the library. The first non-empty allocation creates
one native context with one nonblocking stream; each native operation selects
the device on its calling thread. An unused `Synchronize`, and operations using
only empty tensors, work without CUDA. An empty download still waits if earlier
work exists on the backend.

Our stream orders zero-initialization, transfers and kernels. Add checks the
launch status and returns without waiting for GPU execution. Download and
Synchronize wait for all earlier work on that stream. Separate backend
instances own separate streams and tensors, even on the same GPU; the backend
does not synchronize the whole device. CUDA's [stream API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html)
documents this ordering and synchronization model.

The current transfer path deliberately uses managed staging arrays. For F32:

- Upload copies the supplied `ReadOnlyMemory<single>` to an array, marshals it
  into the native call, and waits until the GPU copy finishes. Callers may reuse
  their original buffer immediately after return.
- Download allocates an array, copies device data into it, waits, then copies
  into the caller's `Memory<single>`. This supports memory slices and keeps the
  caller's buffer unchanged if validation or the device transfer fails.

For F16, the public overloads take `ReadOnlyMemory<Half>` and `Memory<Half>`.
A safe `MemoryMarshal.Cast` copies their raw bits through a `uint16[]` staging
array, which P/Invoke can marshal as a primitive array. The device allocation
uses `elementCount * sizeof(half)` bytes. There is no intermediate `single[]`,
no device F32 buffer, and no numeric conversion during transfer; even NaN
payloads are preserved. Download validates the tensor encoding before writing.

Each transfer therefore needs an extra host array of `elementCount * 4` bytes
for F32, or `elementCount * 2` for F16, in addition to the caller's buffer and
device allocation (excluding array/allocator overhead). This is a simple safe
interop baseline, not a zero-copy path. It uses no F# unsafe pointers or manual
pinning. Although the native API uses `cudaMemcpyAsync`, the enclosing transfer
waits before returning so the marshalled host pointer cannot outlive its call.
See the CUDA [memory API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html)
for allocation and copy semantics.

Use `use`/`Dispose` for backends and tensors. Tensor disposal waits before
freeing its allocation; backend disposal waits and releases every remaining
allocation and its stream. Metadata remains readable after disposal. A
`SafeHandle` also owns the native context as a finalization safety net, but
deterministic disposal avoids retaining VRAM until garbage collection.
Explicit backend disposal reports native synchronization/free errors after
attempting every release. Cleanup makes a best effort after device failure, and
finalization cannot report failures to the caller. There is no `cudaDeviceReset`: disposing one
backend must not invalidate another backend's resources.

The native allocation registry provides additional length, format and ownership
checks at the C ABI boundary, and records the allocated byte size used for
copies. Add dispatches from this recorded format; upload/download use distinct
F32 and F16 entry points and reject a mismatched allocation. It is not a
thread-safety mechanism. As specified
by Core, concurrent calls on one backend, including disposal, are unsupported.
Shapes must match exactly: no broadcasting or strided views. Tensors belong to
their creating instance and cannot be passed to another CUDA or CPU backend.

## Upload stored weights and release VRAM

The goal is to place a matrix on the GPU without expanding its quantized values.
Build the native backend as above, then run this example; it creates its own
36-byte fixture and requires no model installation:

```powershell
dotnet fsi examples/CudaWeights.fsx
```

Expected output: `example.weight: Q4_0, 2 x 32, Cuda`.
The example reads two rows of Q4_0 through Core's `FileTensorSource`, calls
`WeightMatrix.Read`, uploads the result, closes the source and deletes the file.
The returned `IWeightTensor` still owns its independent device copy.

```mermaid
flowchart LR
    Source["Validated stored payload<br/>B bytes, copied or memory-mapped"]
    Staging["Managed staging<br/>min(B, 4 MiB) bytes"]
    Device["Owned VRAM<br/>B bytes, original encoding"]
    Release["Dispose weight or backend<br/>wait for stream, then free"]
    Source -->|copy next byte range| Staging
    Staging -->|transfer and wait before reuse| Device
    Device --> Release
```

`UploadWeights` supports F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0,
IQ4_NL and Q2_K through Q6_K. Rows and columns keep Core's row-major order.
Every byte, including scale bits and non-finite representations, is preserved;
no decoding, numerical conversion or repacking occurs. `WeightMatrix.Read`
has already checked the shape, row block boundaries and payload length.
An upload finishes before returning, so its source need only remain valid and
unchanged for the duration of the call. This also applies to mapped GGUF sources.

A single staging array is reused for successive ranges of at most 4 MiB.
Range boundaries may split quantization blocks: the device receives contiguous
bytes at their original offsets, so block interpretation is unaffected. Native
checks reject negative offsets, overflowing/out-of-bounds ranges, foreign pointers
and execution-tensor allocations before copying. Stored weights cannot be used as
execution tensors, even when their encoding is F32 or F16.

VRAM use is exactly the stored payload size, excluding allocator overhead. Host
staging is bounded independently of that size, rather than duplicating an entire
large matrix. Each range waits on the stream before reusing its staging array.
If a later range fails, upload releases the partially populated allocation and
rethrows the original error. The tensor is only returned after all ranges succeed.

Weight disposal waits for preceding work and frees its allocation. Backend disposal
waits and releases every remaining allocation, including weights whose callers did
not dispose them. Both are idempotent; name, encoding and dimensions remain readable.
Explicit cleanup reports CUDA errors, while the `SafeHandle` finalizer remains a
best-effort fallback. A failed free stays in the native registry for a final cleanup
attempt when the backend closes. Other backend instances remain usable.

Storage types live in `CudaStorage.fs`, `CudaTensor.fs` and `CudaWeightTensor.fs`;
the backend owns their lifetimes and the session owns native interop. Internal raw
readback and byte accounting let the tests verify device contents and releases
without adding a public pointer or weight-download API. Matrix products and
embedding lookup remain separate roadmap items.

## Compare VRAM and mapped host weights (Q.3)

Q.3 is an optional storage alternative, evaluated separately from inference.
The experiment registers an entire read-only file view with
`cudaHostRegisterMapped | cudaHostRegisterReadOnly`, obtains its device address
and reads stored bytes directly from GPU code. This avoids a VRAM weight copy.
The view stays open and registered until all GPU work finishes. Device capability
checks and read-only registration requirements follow NVIDIA's
[CUDA 12.8 memory API](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-runtime-api/group__CUDART__MEMORY.html).
Unified addressing alone does not establish that an arbitrary file mapping is
accessible to GPU code; this experiment explicitly registers it.

To reproduce the comparison on Windows, create a new fixture, then build the
standalone probe. The generated file has eight Q4_K matrices of 155,325 × 3,072
and one 3,072-element F32 vector: the same payload workload as
[A.2](gguf.md#memory-mapped-tensor-source). Its synthetic names produce a 512-byte
header and a total size of 2,147,225,600 bytes. The fixture contains arbitrary
stored bytes for transfer measurements, not meaningful model weights.

```powershell
New-Item -ItemType Directory -Force artifacts/cuda-weight-storage | Out-Null
dotnet fsi examples/CudaWeights.fsx --create-benchmark artifacts/cuda-weight-storage/weights.gguf
cmake -S examples/cuda-weight-storage -B artifacts/cuda-weight-storage/build -G "Visual Studio 17 2022" -A x64 -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build artifacts/cuda-weight-storage/build --config Release
& artifacts/cuda-weight-storage/build/Release/cuda_weight_storage.exe artifacts/cuda-weight-storage/weights.gguf 512 268401600 12288
dotnet fsi examples/CudaWeights.fsx --measure artifacts/cuda-weight-storage/weights.gguf
```

The generator refuses to overwrite an existing file. The probe accepts the data
offset, matrix byte count and vector byte count printed by the generator; it
validates their ranges without implementing another GGUF parser. The managed
measurement uses `GgufTensorSource.openMappedFile` and the actual backend.
On Linux, omit the Visual Studio generator and `-A x64`, set
`-DCMAKE_BUILD_TYPE=Release`, and run `build/cuda_weight_storage`; that path has
not been exercised here.

Measured on 2026-11-21, Windows x64, RTX 4070 Ti SUPER (16 GiB), driver 610.88,
CUDA 12.8 and MSVC 19.42. The native probe is Release, with warm file pages and
five trials. Each trial reports placement, the first full GPU read, then the mean
of ten repeated full reads. The table contains the median across trials:

| Storage path | Placement | First GPU read | Repeated GPU read | Host staging | Weight VRAM |
| --- | ---: | ---: | ---: | ---: | ---: |
| Copy to VRAM, whole-matrix staging | 251 ms | 6.63 ms | 6.56 ms | 256 MiB | 2.0 GiB |
| Copy to VRAM, bounded staging | 283 ms | 6.60 ms | 6.56 ms | 4 MiB | 2.0 GiB |
| Registered read-only mapping | 0.08 ms | 200 ms | 112 ms | None | None |

All paths produced the same full-byte checksum, `268386036982`. File-page warming
also touches the normalization vector; GPU scans cover the eight matrices. The
checksum kernel and its small result buffer are identical for all paths. These
are storage/read measurements, not decoding, matrix multiplication or token rates.
Whole-matrix and bounded native staging each reuse one buffer; their allocation
behavior is not a measurement of the .NET garbage collector.

The actual managed Debug backend uploaded all eight matrices in a median 764 ms
over five warm-cache trials, allocating 32.02 MiB on the calling thread in total
(eight 4 MiB staging arrays plus small objects). Median index/mapping setup was
0.64 ms. These separate managed timings include interop and the Debug native
runtime; they are not directly comparable to the Release native probe.

Bounded staging trades about 13% placement time in the native probe for a 64-fold
reduction in peak staging size. It preserves the VRAM read rate and requires no
new unsafe managed code. Mapped host placement is fast, but repeated reads were
about 17 times slower and the first read included substantial deferred work.
Accordingly the production backend keeps VRAM copies. Q.3 remains deferred;
the reproducible probe is retained to revisit devices or workloads where saved
VRAM matters more than repeated weight-read speed. No automatic fallback or
experimental mapping mode is exposed by `CudaBackend`.

## Study partial RAM/VRAM placement for MoE (Q.4)

The planned Q.4 study asks whether a model whose weights exceed VRAM capacity can
run at useful speed with some experts in RAM. Sparse mixture-of-experts (MoE)
layers select only a subset of experts for each token, but that selection can
change between tokens and layers. For example, the
[Mixtral paper](https://arxiv.org/abs/2401.04088) describes two selected experts
out of eight per layer. A prompt batch can collectively select many more experts
than one token. The dense full-read timings above therefore do not establish the
cost of partial expert placement. Expert offloading is also studied in
[Fast Inference of Mixture-of-Experts Language Models with Offloading](https://arxiv.org/abs/2312.17238);
these references motivate an experiment, not a performance claim for Marmotte.

This is future work: Marmotte currently executes dense `mistral3` only. First
define and validate a chosen MoE architecture's router, expert tensor layout and
weighted output combination against a small independent CPU reference. Use
synthetic fixtures without a model download. CUDA inference kernels are a
prerequisite for end-to-end GPU results; storage-only probes can run earlier.

Compare three strategies under the same explicit memory budget, keeping the
non-expert weights and frequently reused experts in VRAM where the budget permits:

- Read selected RAM-resident experts directly on the GPU through registered
  mappings, extending the Q.3 experiment to sparse access.
- Copy selected experts into a bounded VRAM cache on demand. Compare fixed
  placement with cache replacement, then evaluate prefetching without changing
  the experts selected by the router.
- Execute RAM-resident experts on the CPU and transfer their inputs and outputs,
  measuring synchronization and combination costs as well as expert computation.

For each strategy, preserve stored weight encodings, routing choices and routing
weights; verify outputs against the CPU reference within documented tolerances.
Budget activations, the KV cache and transfer buffers alongside weights. Bound
pinned RAM explicitly and keep mappings, registrations and cached expert buffers
alive until their submitted work completes. Disk-backed paging is a separate
experiment, not an assumed extension of RAM capacity.

Measure prefill and single-token decode separately, with cold and warm expert
caches. Vary expert count, selected experts per token, batch size and VRAM budget;
include uniform selection, repeated selections and abrupt changes in selected
experts. Report total versus active weight bytes, cache hit/miss rates, actual
PCIe bytes per token, transfer/compute time, token latency distributions and
throughput, plus peak RAM, pinned RAM and VRAM. Compare with CPU-only execution
and an all-VRAM baseline where it fits, using identical inputs and encodings.
Keep real-model checks optional and distinguish them from synthetic measurements.
Use these results to decide whether any placement policy merits production support.

## Run the tests

Normal tests require neither CUDA nor a native library:

```powershell
dotnet build Marmotte.slnx
dotnet test Marmotte.slnx --no-build
```

They cover driver-free construction, empty tensors, argument validation,
ownership, disposal and the explicitly unfinished operations. GPU tests are reported as skipped by default. After building and deploying the
native library with `MarmotteBuildCuda=true` or `MarmotteCudaNativePath`, opt in explicitly:

```powershell
$env:MARMOTTE_TEST_CUDA = '1'
dotnet test tests/Marmotte.Backends.Tests/Marmotte.Backends.Tests.fsproj --no-build --filter Category=CUDA
Remove-Item Env:MARMOTTE_TEST_CUDA
```

On Linux, prefix the test command with `MARMOTTE_TEST_CUDA=1`. Opting in makes
missing libraries, incompatible drivers and missing GPUs fail the tests;
they are not silently treated as successful or skipped.

[`CudaTests.fs`](../tests/Marmotte.Backends.Tests/CudaTests.fs) checks CPU agreement,
scalars, matrices, empty tensors, 255/256/257 elements, 65,537 elements, special
F32 values, zero initialization, all exact aliasing cases, queued dependencies,
memory slices, validation before writes, foreign/disposed storage, and disposal
while work is pending. The example and GPU tests were exercised on the Windows
setup above. No model files are required. These are correctness checks, not
performance measurements; no speedup is claimed.

[`HalfTests.fs`](../tests/Marmotte.Backends.Tests/HalfTests.fs) adds the same
independent binary16 fixtures on CPU and CUDA: halfway rounding, overflow,
normal/subnormal transitions, signed zero, infinities and NaNs. It checks all
65,536 bit patterns across upload/download, mixed-format rejection, memory
slices, in-place operations, queued dependencies and lifetime validation.
F16 SwiGLU has no CPU counterpart, so its tests run on CUDA only: exact bits for
independently computed fixtures far from a rounding midpoint, and at most one
unit in the last place from a double-precision reference for all 65,536 gate bit
patterns. The empty F16 CUDA cases still need no native library or device.

[`CudaWeightTests.fs`](../tests/Marmotte.Backends.Tests/CudaWeightTests.fs) checks
bit-for-bit device storage for all fourteen encodings, host slices, mutation after
upload, closed mapped sources, staging boundaries inside quantization blocks,
native byte accounting, partial-upload cleanup, ownership and stream-ordered
disposal. Direct C ABI checks reject overflowing ranges and attempts to pass
stored weights to Add. The lifetime state tests also run without CUDA.

## Diagnose common failures

| Symptom | What to check |
| --- | --- |
| `nvcc` is not found | Toolkit installation and PATH; a working `nvidia-smi` alone is insufficient. |
| CMake cannot find CUDA or a host compiler | CUDA Visual Studio integration, C++ workload, selected generator and Toolkit/compiler compatibility. Use a new build directory after changing toolchains. |
| Unsupported host compiler | Select a supported compiler for the installed Toolkit instead of bypassing the compiler check. |
| `DllNotFoundException` | Build with `MarmotteBuildCuda=true` or `1`, or supply a prebuilt library via `MarmotteCudaNativePath`. Check that the native library reaches the test output folder, and check native dependencies too. For Rider, see [build and test settings](#build-and-run-cuda-tests-in-rider). |
| `BadImageFormatException` | Match the native binary and .NET process architecture; this backend requires 64-bit execution. |
| `EntryPointNotFoundException` | A stale native library was deployed; rebuild it and copy it through the managed build again. |
| No device, insufficient driver, invalid device ordinal | Check `nvidia-smi`, driver compatibility and the chosen `deviceIndex`. |
| No kernel image for this device | Rebuild with an architecture supported by both the GPU and Toolkit. |
| `OutOfMemoryException` | Device allocation failed or the host staging array could not be allocated. Dispose unused tensors and reduce sizes. |
| An error appears at Download or Synchronize | An asynchronous kernel may have failed earlier. Add a temporary Synchronize after the suspected launch to locate it. |
| `NotImplementedException` from another kernel | Only Add and SwiGLU are implemented on CUDA at this stage. |

CUDA runtime failures include the operation, numeric error code and native
message in `InvalidOperationException`; `cudaErrorMemoryAllocation` (status 2)
maps to `OutOfMemoryException`, as specified in the
[runtime error identifiers](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html).
There is no automatic CPU fallback.

## Kernel formats and file layout

Each operation has one `.cu` file containing its shared host entry point and
separate typed GPU functions for its supported storage formats:

| Function | Format | CUDA type | .NET type | Bytes per value |
| --- | --- | --- | --- | --- |
| `addValuesF32` | 32-bit floating point | `float` | `System.Single` (`single` / `float32` in F#) | 4 |
| `addValuesF16` | 16-bit floating point | `half` | `System.Half` | 2 |

**F16 and Half name the same format.** CUDA provides `half` in `cuda_fp16.h`.
The CUDA `half2` type packs two half values; it is a possible optimization
inside an F16 implementation, not another tensor format. See NVIDIA's
[half-precision types](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__HALF.html).
In F#, `float` means a 64-bit value, so the current F32 API uses `single`.

```text
native/
  CMakeLists.txt        Explicit source list and compiler settings
  CudaContext.h        Shared device, stream and allocation state
  CudaRuntime.cu       Allocation, transfers and synchronization
  kernels/
    README.md          Shared developer guide and kernel contracts
    Add.cu             addValuesF32, addValuesF16 and marmotte_cuda_add
    SwiGlu.cu          swiGluF32, swiGluF16 and marmotte_cuda_swiglu
```

Both Add variants are implemented and available through `ITensorBackend`, and so
are both SwiGLU variants. SwiGLU computes
`destination[i] = silu(gate[i]) × up[i]` with the device `expf`; the F16 variant
widens its inputs to F32, applies the same formula and rounds once to F16, so it
behaves like the F32 kernel followed by a single conversion. The CPU backend
implements SwiGLU in F32 only and rejects F16 with `NotSupportedException`.
`Allocate(shape)` remains F32; `Allocate(shape, TensorEncoding.F16)` creates
zero-initialized Half storage. Uploading `ReadOnlyMemory<Half>` also creates F16
storage, and downloading it requires `Memory<Half>`. All three tensors passed
to Add or SwiGLU must have the same format, including empty tensors. Other allocation
formats are rejected; there is no implicit F32/F16 conversion.

The F16 GPU function uses `__hadd` with `half*` inputs and output, rounding to
nearest with ties to even. Storage and transfers stay at two bytes per element.
The matching CPU implementation owns `Half[]`, although individual additions
can use scalar F32 arithmetic before rounding back to Half. See
[Half Add and memory use](kernels/add.md#half-add-and-memory-use) for a shared F#
example, numerical behavior and the official arithmetic references.

The F# project shows native sources and the developer guide. CMake compiles
only sources explicitly listed in `native/CMakeLists.txt`, so a new experimental
file is not silently included in the library. A new operation needs no separate
header, CMake file or directory unless its complexity later justifies them.

## Add another kernel

Follow the [kernel development guide](../src/Marmotte.Cuda/native/kernels/README.md)
and use [Add.cu](../src/Marmotte.Cuda/native/kernels/Add.cu) as the working
example. Create `kernels/<Operation>.cu` with private typed GPU functions and
one shared exported entry point. List it in `native/CMakeLists.txt`; reject
any unsupported format explicitly and document its status in the guide.

Reuse the context for device selection, allocation checks and stream ordering.
Add a private P/Invoke binding in `CudaNative.fs`, then implement the matching
operation in `CudaBackend.fs`, keeping public argument checks there. Keep
transfers in the runtime so intermediate tensors can remain on the GPU.
Document storage, accumulation, rounding and aliasing for each supported
precision, and verify those contracts with independent expected values in the
optional CUDA tests. Build through `MarmotteBuildCuda=true` as shown above.
