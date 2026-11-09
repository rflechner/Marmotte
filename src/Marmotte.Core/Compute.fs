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

/// An owned, writable execution tensor, currently contiguous F32 or F16. Model weights use
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

/// Keys and values of past positions, stored on a backend in the encodings of a
/// KeyValueCacheLayout. Each layer stores consecutive positions from 0; IKeyValueCacheKernel
/// writes and reads them. Disposal follows ITensor's rules. Layout and Backend remain readable
/// after disposal.
type IKeyValueCache =
    inherit IDisposable

    abstract Layout: KeyValueCacheLayout
    abstract Backend: BackendKind

    /// Positions stored in a layer, from position 0, once the store operations submitted so far
    /// complete; a backend tracks this count when they are submitted, without waiting for them.
    /// Raises ArgumentOutOfRangeException for a layer outside [0, Layers) and
    /// ObjectDisposedException after disposal of the cache or its backend.
    abstract StoredPositions: layer: int -> int

/// Element-wise F32 or F16 addition: destination[i] = left[i] + right[i].
type IAddKernel =
    /// All tensors must belong to the executing backend instance and have exactly the same shape
    /// and encoding. F16 stores two bytes per element and rounds each sum to the nearest F16,
    /// ties to even. Overflow produces infinity; subnormals and signed zero are supported.
    /// NaN payloads are not specified. CPU may use scalar F32 arithmetic without F32 tensor buffers.
    /// No broadcasting, strided views, implicit transfers or encoding conversion. Destination may
    /// be either input (or both); otherwise inputs are unchanged. Empty tensors are valid no-ops.
    /// Arguments are validated before submission: ArgumentNullException for null, ArgumentException
    /// for incompatible tensors, ObjectDisposedException for disposed storage or backend.
    /// The result follows ITensorBackend's ordering rules and may be computed after the call returns.
    abstract Add: left: ITensor * right: ITensor * destination: ITensor -> unit

    /// Adds an F16 tensor into an F32 tensor, such as F16 projection output into an F32 residual
    /// stream: destination[i] = left[i] + right[i]. Left and destination are F32 and right is F16;
    /// Add stays strict about encodings, so this widening is always requested by name. Every F16
    /// value converts exactly to F32, and the sum is ordinary F32 addition, rounded to nearest,
    /// ties to even. Destination may be left. Ownership, shapes, validation and ordering follow
    /// Add; any other combination of encodings raises ArgumentException before submission.
    abstract AddF32F16: left: ITensor * right: ITensor * destination: ITensor -> unit

/// Element-wise SwiGLU activation of a gated feed-forward layer:
/// destination[i] = silu(gate[i]) × up[i], with silu(x) = x / (1 + e^-x).
/// See "GLU Variants Improve Transformer", https://arxiv.org/abs/2002.05202.
type ISwiGluKernel =
    /// Currently requires F32 tensors; F16 raises NotSupportedException on CPU.
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
    /// Currently requires F32 tensors; F16 raises NotSupportedException on CPU.
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
    /// weights stay unchanged. CPU supports every TensorEncoding.
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

/// Copy consecutive rows without transferring tensor contents to the host.
type IRowCopyKernel =
    /// Source and destination are rank-2 F32 tensors with the same column count.
    /// Copies destination.Rows rows starting at firstRow, validating the full range
    /// before writing. Empty copies allow firstRow = source.Rows. Identity copies
    /// are valid. Ownership, disposal and submission ordering follow IAddKernel.Add.
    /// Invalid shapes raise ArgumentException; invalid ranges raise ArgumentOutOfRangeException.
    abstract CopyRows: source: ITensor * firstRow: int * destination: ITensor -> unit

/// Matrix-vector product with weights kept in their stored encoding:
/// destination[r] = Σ weights[r, c] × input[c] over the Columns values c of row r.
type IMatVecKernel =
    /// Input has shape [Columns] and destination [Rows], both F32. Every tensor must belong to the
    /// executing backend instance, and destination must not be input, because each output reads
    /// the whole input. Ownership, validation and ordering otherwise follow IAddKernel.Add.
    /// Each output sums Columns products, and implementations decode blocks and sum in different
    /// orders: compare results with a tolerance relative to Σ |weights[r, c] × input[c]|, not to
    /// the result, which cancellation can make small. Raises NotSupportedException for a weight
    /// encoding this backend cannot multiply. A backend may expose an explicit arithmetic
    /// policy that quantizes activations; its documented quantization error is additional
    /// to summation error. Such a policy must not mutate the caller's F32 input.
    abstract MatVec: weights: IWeightTensor * input: ITensor * destination: ITensor -> unit

/// Matrix product of several tokens with the same stored weights, for prompt processing:
/// destination[t, r] = Σ weights[r, c] × input[t, c], which multiplies every input row by the
/// transposed weight matrix. Each token's output is what MatVec gives for its input row, but an
/// implementation can decode each weight row once for the whole batch instead of once per token.
type IMatMulKernel =
    /// Input has shape [Tokens; Columns] and destination [Tokens; Rows], both F32, with Tokens
    /// zero for an empty, no-op batch. Every tensor must belong to the executing backend
    /// instance, and destination must not be input, because each output row reads its whole input
    /// row. Ownership, validation, ordering and tolerances otherwise follow IMatVecKernel.MatVec.
    abstract MatMul: weights: IWeightTensor * input: ITensor * destination: ITensor -> unit

/// Rotary position embedding (RoPE): rotates pairs of values in every head of a token by angles
/// proportional to its position, so that attention scores between queries and keys depend on their
/// relative positions. Models apply it to queries and keys, not values.
type IRopeKernel =
    /// Currently requires F32 tensors; F16 raises NotSupportedException on CPU.
    /// Input and destination have the same shape, of rank 2 or more. The first dimension counts
    /// tokens, and each token's values are consecutive heads of embedding.HeadDimension values, so
    /// both [tokens; heads × HeadDimension] and [tokens; heads; HeadDimension] are accepted.
    /// positions holds each token's absolute position, at least 0, and is read before the call
    /// returns. A pair (a, b) at angle θ becomes (m × (a cos θ - b sin θ), m × (a sin θ + b cos θ)),
    /// with m = embedding.Magnitude and the pairing of the embedding. Destination may be input.
    /// Ownership, validation and ordering follow IAddKernel.Add; positions.Length must equal the
    /// token count, and a negative position raises ArgumentOutOfRangeException. Implementations
    /// approximate sines and cosines differently, so compare results with a tolerance.
    abstract Rope: input: ITensor * positions: ReadOnlyMemory<int> * embedding: RotaryEmbedding * destination: ITensor -> unit

/// Attention temperature: multiplies every query value of a token, after RoPE, by
/// AttentionTemperature.Factor of the token's position. Keys and values are not scaled.
type IAttentionTemperatureKernel =
    /// Currently requires F32 tensors; F16 raises NotSupportedException on CPU.
    /// Queries and destination have the same shape, of rank 2 or more, whose first dimension
    /// counts tokens: [tokens; heads × HeadDimension] or [tokens; heads; HeadDimension].
    /// positions holds each token's absolute position, at least 0, and is read before the call
    /// returns. Destination may be queries. Ownership, validation and ordering follow
    /// IAddKernel.Add; positions.Length must equal the token count, and a negative position raises
    /// ArgumentOutOfRangeException. Each value is multiplied by the F32 factor once, so positions
    /// below FloorLength leave queries unchanged.
    abstract ApplyAttentionTemperature: queries: ITensor * positions: ReadOnlyMemory<int> * temperature: AttentionTemperature * destination: ITensor -> unit

/// Stores the keys and values of new positions in a key-value cache, in the layout's encodings,
/// and decodes stored positions back into F32 tensors.
///
/// Both operations take F32 keys and values tensors of the same shape, whose first dimension counts
/// tokens: [tokens; KeyValueHeads × HeadDimension] or [tokens; KeyValueHeads; HeadDimension]. Token
/// t is position position + t of the layer. Every tensor and the cache must belong to the executing
/// backend instance. All arguments are validated before submission or any write:
/// ArgumentNullException for null, ArgumentException for incompatible tensors or a foreign cache,
/// ArgumentOutOfRangeException for a layer or positions outside the ranges below, and
/// ObjectDisposedException for disposed storage or backend. Ordering follows IAddKernel.Add.
type IKeyValueCacheKernel =
    /// Encodes keys, after RoPE, and values into positions [position, position + tokens) of a
    /// layer, with the layout's KeyEncoding and ValueEncoding; the tensors are unchanged. position
    /// must be in [0, StoredPositions(layer)], so that stored positions stay contiguous, and
    /// position + tokens must not exceed Capacity. Storing below StoredPositions replaces those
    /// positions and discards every later one of the layer, which was computed from the replaced
    /// tokens: StoredPositions(layer) becomes position + tokens, even for an empty batch, which
    /// then only truncates the layer. Keys and values may be the same tensor. See RowEncoder for
    /// the rounding of each encoding and its handling of non-finite values.
    abstract StoreKeyValues: cache: IKeyValueCache * layer: int * position: int * keys: ITensor * values: ITensor -> unit

    /// Decodes stored positions [position, position + tokens) of a layer into keys and values,
    /// which must be different tensors. position must be at least 0 and position + tokens at most
    /// StoredPositions(layer). An F32 cache returns the stored values bit for bit; other encodings
    /// return what RowDecoder decodes from the stored rows, so a stored row always reads back the
    /// same. Reading an empty batch is a valid no-op.
    abstract ReadKeyValues: cache: IKeyValueCache * layer: int * position: int * keys: ITensor * values: ITensor -> unit

/// Causal grouped-query attention over the stored keys and values of one layer.
type IAttentionKernel =
    /// Queries and destination have identical F32 shapes: [tokens; queryHeads; HeadDimension]
    /// or [tokens; queryHeads × HeadDimension]. queryHeads must be positive and divisible by
    /// the cache's KeyValueHeads. Consecutive groups of queryHeads / KeyValueHeads query heads
    /// share one KV head. Queries already include RoPE and any attention temperature scaling.
    /// Token t attends only to positions 0 through position + t, inclusive. All of these positions
    /// must already be stored in the chosen layer. position is in [0, StoredPositions(layer)];
    /// an empty batch is a no-op. Neither queries nor cache is modified; destination must differ
    /// from queries. Scores are dot(Q, K) / sqrt(HeadDimension), normalized by softmax after
    /// subtracting their maximum, then used to sum V. CPU scores use F32; its online
    /// mode uses double-precision running value sums and denominator, while the reference
    /// mode uses F32 reductions. Both decode one cached head at a time. Compare backends
    /// with numerical tolerances. Non-finite inputs or
    /// overflowing scores may produce NaN; masking excludes future positions entirely.
    /// Ownership, validation and ordering follow IKeyValueCacheKernel; invalid head dimensions
    /// or aliasing raise ArgumentException. Validation completes before any destination write.
    abstract Attention: cache: IKeyValueCache * layer: int * position: int * queries: ITensor * destination: ITensor -> unit
