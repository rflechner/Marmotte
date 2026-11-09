# Metal: from the first build to an Add on the GPU

By the end of this guide you will have compiled Marmotte's Metal kernels, run an
addition on your Mac's GPU, and be able to follow that call from F# to the
compute function.

The Metal backend implements contiguous F32 and F16 tensors, zero-initialized
allocation, uploads, downloads, synchronization, element-wise Add, the mixed
`AddF32F16` and SwiGLU in F32 and F16. It uses native
Objective-C++ and Metal Shading Language (MSL) through P/Invoke, with no
additional NuGet dependency. Other kernels, weight upload and the key-value cache
still throw `NotImplementedException`.

This guide assumes familiarity with building the .NET solution, but no Metal
experience. Commands run from the repository root. The native build is optional:
`dotnet build Marmotte.slnx` and the default tests work without it, on any
operating system.

## What the different tools do

| Component | Purpose |
| --- | --- |
| Mac with a Metal GPU | Executes the parallel arithmetic. On Apple silicon, the GPU and CPU share the same memory. |
| Xcode | Provides the macOS SDK, the Metal framework headers and the Objective-C++ compiler. |
| Metal Toolchain | Provides the `metal` command that compiles `.metal` files. Recent versions of Xcode install it separately. |
| CMake 3.24 or later | Compiles the shaders and the host code into one library. |
| .NET SDK | Compiles the F# backend. With `MarmotteBuildMetal=true`, MSBuild also invokes CMake. |

## Step 1: install the prerequisites

Install Xcode from the App Store and open it once to finish its setup. The Command
Line Tools alone are not enough, because they do not include the Metal compiler.
Then install the two remaining tools:

```bash
xcodebuild -downloadComponent MetalToolchain
```

```bash
brew install cmake
```

`xcrun -sdk macosx metal --version` must print a version. If it reports a missing
Metal Toolchain, run the first command above.

## Step 2: build the native library

```bash
dotnet build Marmotte.slnx -p:MarmotteBuildMetal=true
```

MSBuild runs CMake, which runs this pipeline:

```mermaid
flowchart LR
    Shader["kernels/Add.metal"] -->|"metal -fno-fast-math"| Library["marmotte_kernels.metallib"]
    Host["MetalRuntime.mm, kernels/Add.mm"] -->|"clang++ -fobjc-arc"| Objects["Object files"]
    Library -->|"section __TEXT,__metallib"| Dylib["libmarmotte_metal.dylib"]
    Objects --> Dylib
    Dylib -->|"copied by MSBuild"| Output["Beside Marmotte.Metal.dll"]
```

Embedding the compiled shaders in a section of the library means there is one
file to deploy, and no path to find at run time.

| MSBuild property | Default and purpose |
| --- | --- |
| `MarmotteBuildMetal` | `false`; set to `true` or `1` to build the native library before the managed project. Both environment variables and `-p:` properties accept these values. |
| `MarmotteMetalBuildDirectory` | `artifacts/metal/msbuild/<Configuration>` under the repository. |
| `MarmotteMetalNativePath` | Empty. Set it to deploy a prebuilt `libmarmotte_metal.dylib` instead of building one. |

## Step 3: run an Add on the GPU

```bash
dotnet fsi examples/MetalAdd.fsx
```

Expected output:

```text
Metal Add, then in-place Add: [|12.0f; 24.0f; 36.0f; 48.0f; 60.0f; 72.0f|]
Metal Half Add, then in-place Add: [|12; 24; 36; 48; 60; 72|]
Metal AddF32F16 in place: [|24.0f; 48.0f; 72.0f; 96.0f; 120.0f; 144.0f|]
```

The script adds two `[2; 3]` tensors, then adds the left tensor again in place.
For example, the first value is 1 + 10 + 1 = 12. It repeats this in F16, then adds
the F16 result into the F32 result: 12 + 12 = 24.

## Run the tests

Two settings control different stages:

| Setting | Read by | Purpose |
| --- | --- | --- |
| `MarmotteBuildMetal=true` or `1` | MSBuild, during the build | Compiles the native library and copies it beside the test assemblies. |
| `MARMOTTE_TEST_METAL=1` | The test process | Enables the optional GPU tests. Once enabled, a missing library or device fails them. |

```bash
dotnet build tests/Marmotte.Backends.Tests/Marmotte.Backends.Tests.fsproj -p:MarmotteBuildMetal=true
```

```bash
MARMOTTE_TEST_METAL=1 dotnet test tests/Marmotte.Backends.Tests/Marmotte.Backends.Tests.fsproj --no-build --filter "FullyQualifiedName~MetalTests"
```

Without `MARMOTTE_TEST_METAL=1`, tests that need a GPU are skipped. Tests of empty
tensors and argument validation always run, because those paths load no native code.

## Follow one call through the code

| File | Responsibility |
| --- | --- |
| [`MetalBackend.fs`](../src/Marmotte.Metal/MetalBackend.fs) | Tensor shapes, instance ownership, disposal and the public Core interfaces. |
| [`MetalNative.fs`](../src/Marmotte.Metal/MetalNative.fs) | P/Invoke declarations, native error translation and the context `SafeHandle`. |
| [`native/MetalContext.h`](../src/Marmotte.Metal/native/MetalContext.h) | Private shared state: device, command queue, pipelines, buffers and in-flight work. |
| [`native/MetalRuntime.mm`](../src/Marmotte.Metal/native/MetalRuntime.mm) | Device initialization, shader loading, allocation, copies, synchronization and cleanup. |
| [`native/kernels/Add.metal`](../src/Marmotte.Metal/native/kernels/Add.metal) | The F32, F16 and mixed compute functions that run on the GPU. |
| [`native/kernels/Add.mm`](../src/Marmotte.Metal/native/kernels/Add.mm) | `marmotte_metal_add` and `marmotte_metal_add_f32_f16`: native validation, format dispatch, encoding and submission. |
| [`native/kernels/SwiGlu.metal`](../src/Marmotte.Metal/native/kernels/SwiGlu.metal) | The F32 and F16 SwiGLU compute functions: `silu(gate[i]) * up[i]`. |
| [`native/kernels/SwiGlu.mm`](../src/Marmotte.Metal/native/kernels/SwiGlu.mm) | `marmotte_metal_swi_glu`: native validation, encoding and submission. |
| [`native/CMakeLists.txt`](../src/Marmotte.Metal/native/CMakeLists.txt) | Native sources, compiler settings and output location. |

```mermaid
sequenceDiagram
    participant App as F# application
    participant Backend as MetalBackend
    participant Native as Native library
    participant GPU as Metal queue
    App->>Backend: Upload(shape, host values)
    Backend->>Native: marmotte_metal_upload
    Native->>Native: Copy into a new shared buffer
    Backend-->>App: Owned ITensor
    App->>Backend: Add(left, right, destination)
    Backend->>Backend: Validate shapes, owners and lifetimes
    Backend->>Native: marmotte_metal_add
    Native->>GPU: Commit one command buffer
    Backend-->>App: Return after submission
    App->>Backend: Download(destination, host buffer)
    Backend->>Native: marmotte_metal_download
    Native->>GPU: Wait for every submitted command buffer
    Native->>Native: Copy the shared buffer to the host array
    Backend-->>App: Completed host buffer
```

For `[2; 3]`, storage is six consecutive values in row-major order: 24 bytes in
F32 or 12 bytes in F16. Shape
validation stays in F#; the native addition only needs the element count.

`marmotte_metal_add` runs on the CPU. It selects `addValuesF32` or `addValuesF16`
from the encodings in its allocation registry, and rejects mixed formats before
encoding anything. Then it encodes a dispatch of exactly `count`
threads, with
[`dispatchThreads`](https://developer.apple.com/documentation/metal/mtlcomputecommandencoder/dispatchthreads(_:threadsperthreadgroup:)).
Each thread receives its element index as `[[thread_position_in_grid]]`. The
last threadgroup may be smaller than the others. This requires non-uniform
threadgroup support, which the backend checks against the GPU families listed in
Apple's [feature set tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf).
No threads communicate, and each output reads only the same index of each input,
which makes exact in-place Add valid.

## Memory, ordering and lifetime

`new MetalBackend()` uses
[the system's default device](https://developer.apple.com/documentation/metal/mtlcreatesystemdefaultdevice()).
Construction does not load the library. The first non-empty allocation creates
one native context with one command queue. An unused `Synchronize`, and operations
using only empty tensors, work without Metal.

Tensors are `MTLBuffer` objects in shared storage, created with
[`newBufferWithLength`](https://developer.apple.com/documentation/metal/mtldevice/makebuffer(length:options:)),
which zero-fills them. Each `Add` commits its own command buffer and returns.
These rules follow from Metal's behavior:

- **Order.** One queue plus tracked buffers means that a kernel waits for the
  earlier writes it depends on.
- **Backpressure.** A queue accepts a limited number of uncompleted command
  buffers. Beyond that limit, `Add` blocks until the GPU catches up.
- **Disposal.** Command buffers
  [retain the resources they use](https://developer.apple.com/documentation/metal/mtlcommandbuffer/retainedreferences),
  so disposing a tensor does not wait, even when submitted work still uses it.
- **Errors.** A command buffer can fail after `Add` has returned. The backend
  records the first failure and raises it at the next `Download` or `Synchronize`,
  as `InvalidOperationException` or `OutOfMemoryException`. This path has no
  automated test, because a GPU failure cannot be provoked on demand.

`Upload` and `Download` copy through a temporary managed array to avoid unsafe
code, so a transfer briefly uses twice the tensor's size on the host. F16 transfers
copy raw bits through a `uint16[]` array, so they stay at two bytes per element and
preserve all 65,536 bit patterns, without any F32 buffer. Shared storage leaves
room for avoiding these copies later.

## Floating-point behavior

The shaders are compiled with `-fno-fast-math`, because
[Metal enables fast math by default](https://developer.apple.com/documentation/metal/mtlcompileoptions/fastmathenabled).
NaN, infinities and signed zero then follow IEEE 754, and tests cover them.

**F32 subnormal values are the exception.** An Apple M4 Max flushes them to zero,
for both inputs and results:

| Addition | CPU result | Metal on M4 Max |
| --- | --- | --- |
| `1e-45f + 1e-45f` | `0x00000002` | `0x00000000` |
| Largest subnormal + `1e-45f` | `0x00800000` | `0x00000000` |

Compiling with `-std=metal3.2 -fmetal-math-mode=safe` gave the same results. So
when comparing with the CPU reference, use an absolute tolerance of about
`1.2e-38`, the smallest normal F32 value. Transfers are unaffected: they copy
bytes and preserve every bit.

**F16 subnormal values are kept.** The same GPU computes them exactly, and rounds
to nearest, ties to even: F16 Add matches the CPU bit for bit on the fixtures of
[`HalfTests.fs`](../tests/Marmotte.Backends.Tests/HalfTests.fs), including
`0x0001 + 0x0001 = 0x0002`.

`AddF32F16` follows from both rules. Every F16 value, subnormal or not, is a normal
F32 value after widening, so the right operand is exact. The left operand and the
sum are F32, so subnormals there are flushed. That is why a mixed sum is not simply
an F16 sum: `(1 + 3 × 2⁻²³) + 1` rounds in F32 to `2 + 2⁻²¹`, where F16 arithmetic
would give `2`.

## Measurements

Total time per Add, for repeated Adds followed by one `Synchronize`, averaged over
at least 20 Adds. Release builds on an Apple M4 Max (32-core GPU). These numbers
are indicative, not a benchmark suite.

| Elements | CPU backend | Metal backend |
| ---: | ---: | ---: |
| 1 | 0.3 µs | 13.1 µs |
| 1,024 | 1.0 µs | 10.5 µs |
| 1,048,576 | 116 µs | 50 µs |
| 67,108,864 | 7,272 µs | 2,718 µs |

F16 and mixed additions, with the same method:

| Elements | CPU F16 `Add` | Metal F16 `Add` | CPU `AddF32F16` | Metal `AddF32F16` |
| ---: | ---: | ---: | ---: | ---: |
| 1,024 | 10.0 µs | 11.7 µs | 7.2 µs | 8.7 µs |
| 1,048,576 | 2,451 µs | 90 µs | 889 µs | 49 µs |
| 67,108,864 | 156,897 µs | 2,817 µs | 57,160 µs | 2,623 µs |

The CPU loops for these two operations are scalar, while F32 Add is vectorized, so
the gap on large tensors is wider here.

Small additions are dominated by the cost of one command buffer per `Add`, so the
CPU is faster for them. Batching several kernels into one command buffer is the
obvious next step once the forward pass runs on Metal.

## SwiGLU

`SwiGlu(gate, up, destination)` computes `destination[i] = silu(gate[i]) × up[i]`
with `silu(x) = x / (1 + e^-x)`. Metal accepts F32 and F16, with all three tensors
in the same encoding; mixed encodings raise `ArgumentException` before any
submission. (The CPU requires F32 and raises `NotSupportedException` for F16.)
The destination may be `gate` or `up`, because each thread reads and writes only
its own index.

Metal's `exp` is a different approximation from the CPU's, so results may differ
by a few units in the last place: compare F32 with a relative tolerance of about
`1e-6`, and F16 with about `1e-2`, as `MetalTests.fs` does. The F16 function
widens its operands, computes in F32 and rounds once to F16: in binary16, `e^-x`
would overflow below -11.1 and turn representable results such as `silu(-12)`
into zero. The special values follow the formula: `silu(-∞)` is NaN, `silu(+∞)`
is `+∞`, and where `e^-x` overflows in F32 (gate below about -88.7) the result
is a signed zero. The F32 subnormal
flush described above also applies to F32 SwiGLU results, so the tests add an
absolute floor of `1e-30`; F16 subnormals are kept exactly.

## Known limitations

- Only Add, `AddF32F16` and SwiGLU (F32 and F16) run on Metal; every other kernel is a placeholder.
- The default device only: there is no device selection for Macs with several GPUs.
- Tested on Apple silicon only (M4 Max, macOS 27).
- The [kernel development guide](../src/Marmotte.Metal/native/kernels/README.md)
  explains how to add the next operation.
