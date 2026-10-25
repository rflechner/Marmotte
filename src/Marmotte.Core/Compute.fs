namespace Marmotte.Core

open System
open System.Collections.Generic

/// Execution backend family. This identifies a kind of storage, not a device or context instance.
[<RequireQualifiedAccess>]
type BackendKind =
    | Cpu
    | Cuda
    | Metal

/// Immutable row-major dimensions for a contiguous execution tensor, with the last dimension
/// contiguous. An empty shape is a scalar (one element); any zero dimension means no elements.
/// Unlike stored TensorDescriptor shapes, execution shapes currently use managed-array limits.
/// Shapes compare by dimensions, so F# `=` and hashed collections treat [2; 3] and [2; 3] as equal.
[<Sealed>]
type TensorShape private (dimensions: int array, elementCount: int) =
    let readOnlyDimensions = Array.AsReadOnly(dimensions)

    member private _.RawDimensions = dimensions

    /// Read-only dimensions owned by this shape; callers cannot mutate the underlying array.
    member _.Dimensions = readOnlyDimensions :> IReadOnlyList<int>
    member _.Rank = dimensions.Length
    member _.ElementCount = elementCount

    /// Same dimensions in the same order; a [6] vector differs from a [2; 3] matrix.
    member _.Equals(other: TensorShape) =
        not (obj.ReferenceEquals(other, null)) && dimensions = other.RawDimensions

    override this.Equals(other: obj) =
        match other with
        | :? TensorShape as shape -> this.Equals(shape)
        | _ -> false

    override _.GetHashCode() = hash dimensions

    /// F# list notation, such as "[2; 3]", or "[]" for a scalar.
    override _.ToString() =
        let joined = dimensions |> Array.map string |> String.concat "; "
        $"[{joined}]"

    interface IEquatable<TensorShape> with
        member this.Equals(other) = this.Equals(other)

    /// Copies dimensions. Raises ArgumentNullException for null, or ArgumentException for
    /// negative dimensions or a non-empty shape whose element count exceeds Array.MaxLength.
    static member Create(dimensions: int array) =
        ArgumentNullException.ThrowIfNull(dimensions)
        let dimensions = Array.copy dimensions

        if dimensions |> Array.exists (fun dimension -> dimension < 0) then
            invalidArg (nameof dimensions) "Tensor dimensions must be non-negative."

        let elementCount =
            if dimensions |> Array.contains 0 then
                0
            else
                let mutable count = 1
                for dimension in dimensions do
                    if count > Array.MaxLength / dimension then
                        invalidArg (nameof dimensions) "Tensor element count exceeds the managed array limit."
                    count <- count * dimension
                count

        TensorShape(dimensions, elementCount)

/// An owned, writable execution tensor, currently contiguous F32 only. Model weights use
/// IWeightTensor instead. Payload access goes through the creating backend, so device storage
/// need not expose host memory. Disposal follows ITensorBackend's lifetime rules: it is idempotent,
/// ordered after submitted operations that use the tensor, and a no-op once the backend is disposed.
/// Metadata remains readable after disposal.
type ITensor =
    inherit IDisposable

    abstract Shape: TensorShape
    abstract Encoding: TensorEncoding
    abstract Backend: BackendKind

/// Read-only model weights placed on a backend in their stored encoding, never decoded as a whole.
/// As in WeightMatrix, Columns is the contiguous dimension and each row holds whole quantization
/// blocks, so kernels can decode or multiply one row at a time. Disposal follows the same rules
/// as ITensor. Metadata remains readable after disposal.
type IWeightTensor =
    inherit IDisposable

    abstract Name: string
    abstract Encoding: TensorEncoding
    abstract Rows: int
    abstract Columns: int
    abstract Backend: BackendKind

/// Element-wise F32 addition: destination[i] = left[i] + right[i].
type IAddKernel =
    /// All tensors must belong to the executing backend instance and have exactly the same shape.
    /// No broadcasting, strided views, implicit transfers or encoding conversion. Destination may
    /// be either input (or both); otherwise inputs are unchanged. Empty tensors are valid no-ops.
    /// Arguments are validated before submission: ArgumentNullException for null, ArgumentException
    /// for incompatible tensors, ObjectDisposedException for disposed storage or backend.
    /// The result follows ITensorBackend's ordering rules and may be computed after the call returns.
    abstract Add: left: ITensor * right: ITensor * destination: ITensor -> unit

/// Element-wise SwiGLU activation of a gated feed-forward layer:
/// destination[i] = silu(gate[i]) × up[i], with silu(x) = x / (1 + e^-x).
/// See "GLU Variants Improve Transformer", https://arxiv.org/abs/2002.05202.
type ISwiGluKernel =
    /// Tensor rules, validation and ordering are those of IAddKernel.Add, with gate and up as the
    /// inputs. Implementations approximate the exponential differently, so results may differ by a
    /// few units in the last place: compare them with a relative tolerance. Where e^-x overflows
    /// (x below about -88.7), the result is a signed zero instead of a value below 1e-36 in
    /// magnitude, and silu(-∞) is NaN, as the formula gives.
    abstract SwiGlu: gate: ITensor * up: ITensor * destination: ITensor -> unit

/// RMS normalization of each row along the last dimension, followed by a weight per column:
/// destination[r, c] = input[r, c] / sqrt(mean(input[r, *]²) + epsilon) × weight[c].
/// See "Root Mean Square Layer Normalization", https://arxiv.org/abs/1910.07467.
type IRmsNormKernel =
    /// Input and destination have the same shape, of rank 1 or more. Weight is a vector whose
    /// length is the last dimension of input, shared by every row. Epsilon must be finite and
    /// positive, so that an all-zero row stays finite; models store it as
    /// attention.layer_norm_rms_epsilon. Destination may be any input; otherwise inputs are
    /// unchanged. Empty tensors are valid no-ops. Ownership, validation and ordering follow
    /// IAddKernel.Add, and an invalid epsilon raises ArgumentOutOfRangeException.
    /// The mean is a reduction that implementations sum in different orders, so compare results
    /// with a relative tolerance. NaN or infinite values make their row's results NaN or zero,
    /// as the formula gives.
    abstract RmsNorm: input: ITensor * weight: ITensor * epsilon: single * destination: ITensor -> unit

/// Selects vocabulary rows from stored weights and decodes them into F32 activations.
type IEmbeddingKernel =
    /// Writes destination[i, column] = weights[tokenIds[i], column], without scaling or
    /// normalization. Destination must have exactly [tokenIds.Length; weights.Columns] dimensions,
    /// including [1; Columns] for one token and [0; Columns] for an empty, no-op lookup.
    /// IDs are zero-based, must be in [0, weights.Rows), and may repeat or arrive in any order.
    /// Weights and destination must belong to this backend instance. Only selected rows are decoded;
    /// weights stay unchanged. CPU supports F32, F16, BF16, Q4_0, Q8_0, Q4_K and Q6_K.
    /// Floating-point special values follow ordinary F32 conversion and arithmetic.
    ///
    /// All arguments, including every ID, are validated before submission or any destination write:
    /// ArgumentNullException for null tensors, ArgumentException for incompatible tensors,
    /// ArgumentOutOfRangeException for an invalid ID, NotSupportedException for an unsupported
    /// encoding, and ObjectDisposedException for disposed storage or backend.
    /// IDs are consumed or copied before returning, so callers may immediately reuse their buffer.
    /// Do not mutate it during the call. Results follow ITensorBackend's ordering and lifetime rules;
    /// keep the weights' tensor source alive while they are in use.
    abstract EmbeddingLookup: weights: IWeightTensor * tokenIds: ReadOnlyMemory<int> * destination: ITensor -> unit

/// Creates and operates on its own tensors. Separate instances are separate ownership domains,
/// even when their Backend values match.
///
/// Ordering: operations submitted to one instance take effect in submission order, so a kernel
/// observes the results of earlier kernels and transfers. Kernels may return before they complete.
/// Upload and UploadWeights copy or retain their input before returning; Download and Synchronize
/// wait for every previously submitted operation. A device failure in a submitted operation is
/// raised by a later call on the same instance, at the latest by the next Download or Synchronize.
///
/// Lifetime: callers own returned tensors and should dispose them when they are no longer needed.
/// Disposing the backend waits for submitted operations, then releases the storage of every tensor
/// it still owns; those tensors become disposed, and disposing them again is a no-op. Backend
/// disposal is idempotent, and later operations raise ObjectDisposedException.
///
/// Concurrent calls on one instance, including disposal, are not supported. Allocation can fail
/// with OutOfMemoryException. CUDA and Metal placeholders raise NotImplementedException for all
/// operations until implemented.
type ITensorBackend =
    inherit IDisposable
    inherit IAddKernel
    inherit ISwiGluKernel
    inherit IRmsNormKernel
    inherit IEmbeddingKernel

    abstract Backend: BackendKind

    /// Allocates zero-initialized F32 storage. Raises ArgumentNullException for a null shape
    /// or ObjectDisposedException after backend disposal.
    abstract Allocate: shape: TensorShape -> ITensor

    /// Copies exactly Shape.ElementCount host values, in row-major order, into new storage before
    /// returning; the caller may reuse its buffer immediately. Raises ArgumentNullException for a
    /// null shape, ArgumentException for an incorrect source length, or ObjectDisposedException
    /// after backend disposal.
    abstract Upload: shape: TensorShape * values: ReadOnlyMemory<single> -> ITensor

    /// Places a stored matrix on this backend in its encoding, without decoding it. The tensor may
    /// retain the matrix payload instead of copying it, as the CPU backend does, so the tensor
    /// source that produced the matrix must stay alive while the weight tensor is in use.
    /// Raises ArgumentNullException for null, NotSupportedException for an encoding this backend
    /// cannot hold, or ObjectDisposedException after backend disposal.
    abstract UploadWeights: weights: WeightMatrix -> IWeightTensor

    /// Waits for previously submitted operations, then copies the tensor to an exact-length host
    /// buffer in row-major order; the buffer is complete on return. Validates before writing.
    /// Raises ArgumentNullException for null, ArgumentException for a foreign tensor or incorrect
    /// destination length, or ObjectDisposedException for disposed storage or backend.
    abstract Download: tensor: ITensor * destination: Memory<single> -> unit

    /// Waits until every previously submitted operation has completed, and raises any device
    /// failure they reported. Raises ObjectDisposedException after backend disposal.
    abstract Synchronize: unit -> unit
