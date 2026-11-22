# CopyRows: select consecutive token rows

[Back to execution tensors and kernels](../kernels.md).

Learn how to keep a range of activation rows on the backend. Prerequisites:
arrays and [row-major tensor shapes](../kernels.md#reading-the-formulas).
`CopyRows` preserves values exactly; it performs no arithmetic or decoding.

## Follow a two-row copy

Suppose the source has shape `[4; 2]`, with four token rows and two features:

| Source row | Values | Destination row when `firstRow = 1` |
| --- | --- | --- |
| 0 | `[10; 11]` | Outside the selected range |
| 1 | `[20; 21]` | 0 |
| 2 | `[30; 31]` | 1 |
| 3 | `[40; 41]` | Outside the selected range |

A destination of shape `[2; 2]` selects two rows. The caller supplies the
starting row, and the destination's shape determines how many rows to copy:

```text
destination[r, c] = source[firstRow + r, c]
```

Here the flat source offset is `firstRow × columns = 1 × 2 = 2` elements,
and the copy length is `destination.Rows × columns = 2 × 2 = 4` elements.
Source and destination retain separate storage unless they are the same tensor.

## Try it on the CPU

From the repository root, build the CPU library and start F# Interactive:

```sh
dotnet build src/Marmotte.Cpu -c Release
dotnet fsi
```

Paste the following block, including `;;`. No model file is needed. The
download makes the example's result visible; copying rows itself stays on
the backend.

```fsharp
#r "src/Marmotte.Core/bin/Release/net10.0/Marmotte.Core.dll"
#r "src/Marmotte.Cpu/bin/Release/net10.0/Marmotte.Cpu.dll"

open System
open Marmotte.Core
open Marmotte.Cpu

let copyRowsExample () =
    use backend = new CpuBackend() :> ITensorBackend
    let sourceValues = [| 10.0f; 11.0f; 20.0f; 21.0f; 30.0f; 31.0f; 40.0f; 41.0f |]
    use source = backend.Upload(TensorShape.Create [| 4; 2 |], ReadOnlyMemory<single>(sourceValues))
    use destination = backend.Allocate(TensorShape.Create [| 2; 2 |])

    backend.CopyRows(source, 1, destination)

    let result = Array.zeroCreate<single> 4
    backend.Download(destination, result.AsMemory())
    result

printfn "%A" (copyRowsExample ())
;;
```

Expected output: `[|20.0f; 21.0f; 30.0f; 31.0f|]`.

## Why inference needs this operation

When only the last token's logits are requested, `TransformerModel` copies its
hidden-state row into a one-row tensor. Subsequent operations can then work
on the selected row. See [selected-logit execution](../forward-pass.md#request-only-the-logits-you-need).
This uses activations already on the backend; [embedding lookup](embedding.md)
instead selects vocabulary weight rows using token IDs.

## Contract and implementation

[`IRowCopyKernel`](../../src/Marmotte.Core/Compute.fs) requires rank-2 F32 source
and destination tensors with the same column count. Both must be live and
owned by the executing backend instance. The full selected range must fit:
`0 <= firstRow <= source.Rows` and
`destination.Rows <= source.Rows - firstRow`.

An empty destination allows `firstRow = source.Rows`. An identity copy,
`backend.CopyRows(source, 0, source)`, is valid. There are no strided views,
arbitrary row-index lists or implicit precision conversions.

Validation finishes before writing. Incompatible shapes or ownership raise
`ArgumentException`, invalid row ranges raise `ArgumentOutOfRangeException`,
null tensors raise `ArgumentNullException`, and disposed storage or backend
raises `ObjectDisposedException`. CPU F16 input is unsupported and raises
`NotSupportedException`.

The [CPU implementation](../../src/Marmotte.Cpu/CpuBackend.fs) copies one
contiguous span and completes before returning. Its work is proportional to
the copied element count, with no temporary tensor. CUDA and Metal currently
raise `NotImplementedException`. The [backend tests](../../tests/Marmotte.Backends.Tests/BackendTests.fs)
cover selected ranges, identity and empty copies, and validation failures.
