# RMSNorm: normalize the magnitude of each row

[Back to execution tensors and kernels](../kernels.md).

This walkthrough follows one row from its original values to its normalized,
weighted output. You only need arrays, multiplication, averages and square
roots; the runnable example requires the .NET 10 SDK, with no model download.

In a transformer, a row usually contains the features representing one token.
These intermediate values are called **activations**. RMSNorm brings each row
to a comparable magnitude, then applies learned scales to its individual
features. It keeps the same number of rows and columns.

![RMSNorm: square the row [3, 4], average to 12.5, divide the original values by sqrt(12.5 + 0.01), then multiply by the learned weights [2, 0.5] to obtain approximately [1.6964, 0.5655]. Each row has its own magnitude and shares the same weight vector.](rmsnorm.png)

[Open the full-size illustration](rmsnorm.png). Values in the picture are
rounded to four decimals; calculate with unrounded intermediates.

## 1. Measure the row with RMS

**RMS** means *root mean square*. Read the name from right to left to perform
the calculation: square each value, take their mean, then take the square root.
For the row `x = [3; 4]`:

```text
squares     = [3²; 4²] = [9; 16]
meanSquare  = (9 + 16) / 2 = 12.5
RMS(x)      = sqrt(12.5) ≈ 3.5355
```

Squaring prevents positive and negative values from cancelling each other:
`[-3; 4]` has the same RMS as `[3; 4]`. Averaging accounts for the number of
features, and the square root returns to the original values' units. The
result is one measure of magnitude for the whole row.

## 2. Divide the original values by a shared denominator

Marmotte adds a positive constant, `epsilon`, to the mean square **inside**
the square root. This makes the denominator nonzero even for an all-zero row.
Using the illustration's teaching value `epsilon = 0.01`:

```text
denominator = sqrt(meanSquare + epsilon)
            = sqrt(12.5 + 0.01) ≈ 3.5369
normalized  = [3 / denominator; 4 / denominator]
            ≈ [0.8482; 1.1309]
```

Both original values use the same denominator, so their ratio and signs are
preserved. The squared values were only used to measure the row; they do not
replace the input. RMSNorm never subtracts the row's mean.

For a nonzero row, dividing by its exact RMS would give a new RMS of one.
With positive epsilon, the normalized RMS is slightly below one when the mean
square is large compared with epsilon, and smaller when epsilon dominates.
An all-zero row stays zero for finite weights. The model supplies its epsilon;
`0.01` here makes the arithmetic easy to follow, rather than prescribing a
model setting.

## 3. Apply a learned scale to each feature

Normalization removes the row's overall scale. A learned vector, `weight`,
then lets the model adjust each feature separately. Its values were learned
during training and are reused during inference; they are not calculated from
the current row.

For `weight = [2; 0.5]`, multiply matching positions:

```text
output[0] = (3 / sqrt(12.51)) × 2   ≈ 1.6964
output[1] = (4 / sqrt(12.51)) × 0.5 ≈ 0.5655
```

The first feature is amplified and the second attenuated. The output's RMS
need not be one after these learned scales. This kernel adds no bias.

## Repeat independently for each row

For a tensor of shape `[tokens; features]`, the final dimension holds the
features of one row. Each token gets its own denominator; all rows use the
same weight vector. For example, with shape `[2; 2]`, `weight = [2; 0.5]`
and `epsilon = 0.01`:

| Row | Input | Mean square | Denominator | Output, approximately |
| --- | --- | --- | --- | --- |
| 0 | `[3; 4]` | `12.5` | `sqrt(12.51)` | `[1.6964; 0.5655]` |
| 1 | `[0; 0]` | `0` | `sqrt(0.01) = 0.1` | `[0; 0]` |

The zero row does not change the first row's calculation. Higher-rank tensors
follow the same rule: `[2; 3; 4]` contains six independent rows of four values.
A vector `[4]` is one row of four values.

The complete formula uses `r` for the row, `c` for the column, and `C` for
the number of columns. `sum_c` means summing over that row's columns:

```text
meanSquare[r]     = (sum_c input[r, c]²) / C
scale[r]          = 1 / sqrt(meanSquare[r] + epsilon)
destination[r, c] = input[r, c] × scale[r] × weight[c]
```

Computing one number from all the values of a row is called a **reduction**.
It must finish before that row can be scaled. This differs from
[SwiGLU](swiglu.md), where each output depends only on its two matching inputs.

## Try the illustrated calculation in F#

Run `dotnet fsi`, then paste the following block, including the final `;;`.
This small example uses double precision to expose the intermediate values;
Marmotte's CPU storage and output are F32, as described below.

```fsharp
let input = [| 3.0; 4.0 |]
let weight = [| 2.0; 0.5 |]
let epsilon = 0.01

let meanSquare = input |> Array.averageBy (fun value -> value * value)
let denominator = sqrt (meanSquare + epsilon)
let normalized = input |> Array.map (fun value -> value / denominator)
let output = Array.map2 (fun value gain -> value * gain) normalized weight

printfn "mean square = %.4f" meanSquare
printfn "denominator = %.4f" denominator
printfn "normalized = [%.4f; %.4f]" normalized[0] normalized[1]
printfn "output = [%.4f; %.4f]" output[0] output[1]
;;
```

The printed result is:

```text
mean square = 12.5000
denominator = 3.5369
normalized = [0.8482; 1.1309]
output = [1.6964; 0.5655]
```

Try `input = [| -3.0; 4.0 |]`: the denominator is unchanged, and the first
output changes sign. With `weight = [| 1.0; 1.0 |]`, the output is simply the
normalized row. With `input = [| 0.0; 0.0 |]`, the denominator is `0.1` and
both outputs are zero.

## Where Marmotte uses RMSNorm

In the dense `mistral3` forward pass, each transformer block normalizes the
hidden state before attention and before the feed-forward network. A final
normalization precedes the output projection that produces vocabulary logits.
Each of these normalization sites has its own learned vector.
See the [forward-pass walkthrough](../forward-pass.md) for the surrounding flow.

In the documented Ministral example, rows have 3,072 values, epsilon is
`attention.layer_norm_rms_epsilon` (`1e-5`), and the learned vectors are
`attn_norm`, `ffn_norm` and `output_norm`. These dimensions and values belong
to that model; the kernel accepts other compatible row widths and epsilons.

## Backend contract

The call is `backend.RmsNorm(input, weight, epsilon, destination)` through
[`IRmsNormKernel`](../../src/Marmotte.Core/Compute.fs).

| Argument | Requirement |
| --- | --- |
| `input` | Contiguous F32 tensor, rank at least one; rows run along the last dimension |
| `weight` | F32 vector of shape `[columns]`, shared by every row |
| `epsilon` | Finite, strictly positive F32 value, added to the mean square |
| `destination` | F32 tensor with exactly the same shape as `input` |

`weight` is an `ITensor`, not an `IWeightTensor`: the loader decodes these small
normalization vectors to F32. All tensors must be live and belong to the
executing backend instance. The caller owns the destination and its lifetime.

Exact buffer reuse is supported, including
`backend.RmsNorm(input, weight, epsilon, input)`. Destination may also be the
weight tensor when the shape requirements permit it. Otherwise the inputs
remain unchanged. Empty tensors are valid no-ops after validation, including
`[0; C]` with weights `[C]` and `[R; 0]` with weights `[0]`; a rank-zero scalar
is rejected. There are no implicit transfers or precision conversions.

Arguments are checked before any writes. Null tensors raise
`ArgumentNullException`; incompatible shapes or ownership raise
`ArgumentException`; disposed tensors or backend raise
`ObjectDisposedException`; invalid epsilon raises `ArgumentOutOfRangeException`.
The CPU implementation requires F32 and rejects F16 with `NotSupportedException`.
CPU work completes before the call returns. CUDA and Metal accept F32 or F16,
with all three tensors in the same encoding, and may complete asynchronously.
Their F16 paths widen values to F32 and round each output once to F16.

## CPU numerical and performance notes

[`CpuKernels.fs`](../../src/Marmotte.Cpu/CpuKernels.fs) performs two passes per
row: measure its sum of squares, then scale its values. This requires O(C)
work per row and a fixed amount of temporary storage, without allocating a
normalized copy of the tensor. Completing the reduction before writing also
allows the destination to be the input.

The CPU widens F32 values to double precision **before squaring** and sums
using two vector accumulators, followed by a scalar tail. This keeps long
rows accurate and avoids F32 overflow when squaring large finite inputs such
as `1e30`. The reciprocal square root is also computed in double precision
and rounded once to F32. Each output then evaluates
`input × scale × weight` with two F32 multiplications.

Summation order and rounding can differ across implementations, so comparisons
use a tolerance. Previously recorded measurements over 1,000 random rows of
3,072 values found a largest relative error of `1.6e-7` against a
double-precision reference; the backend tests use a relative tolerance of
`1e-6`. NaN input contaminates its row's reduction. Infinite input gives a zero
scale, so infinite entries become NaN while finite entries become zero for
finite weights. Other rows remain independent; non-finite weights follow
ordinary floating-point multiplication.

The existing single-threaded Release measurement on an Apple M4 Max is about
`0.73 µs` for a row of 3,072 values, versus `3.3 µs` for the same algorithm
without vectors. For the measured model, 53 normalizations per token take
about `39 µs`. These are recorded measurements, not a new benchmark or a
guarantee for other machines.

## GPU and mixed-precision normalization

CUDA reduces each row in double precision, with one block per row. F16
inputs widen to F32 before the reduction and each result rounds once to F16;
see the [CUDA guide](../cuda.md#kernel-formats-and-file-layout).

`RmsNormF32ToF16(input, weight, epsilon, destination)` is implemented on CPU and Metal. It takes F32 `input` and
`weight` and writes an F16 `destination` of the same shape, for example to store
an activation in two bytes per value. `RmsNorm` stays strict about encodings, so
this narrowing is requested by name, like `AddF32F16`. Each result is computed and
rounded in F32 exactly as by `RmsNorm`, then rounded to the nearest F16, ties to
even; results beyond the F16 range become infinities. The CPU normalizes one row
into an F32 scratch row of `columns` values, then converts it.

Metal normalizes F32 and F16 rows with one GPU thread per row, summing the squares
in F32 rather than double precision; see the
[Metal guide](../metal.md#rms-normalization) for its tolerances and its F32 range.

## References

- [Root Mean Square Layer Normalization, Zhang and Sennrich (2019), section 4](https://arxiv.org/html/1910.07467v1#S4): definition of RMSNorm and its learned feature scales. Marmotte's contract specifies epsilon and floating-point behavior.
- [CPU backend validation](../../src/Marmotte.Cpu/CpuBackend.fs): shape, ownership, encoding and epsilon checks.
- [Backend tests](../../tests/Marmotte.Backends.Tests/BackendTests.fs): independent double-precision reference, row widths, higher ranks, buffer reuse, zero and extreme rows, and validation failures.
