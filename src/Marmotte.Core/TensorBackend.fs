namespace Marmotte.Core

open System

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
/// with OutOfMemoryException. Unimplemented backend operations raise NotImplementedException.
/// Implemented operations reject unsupported tensor encodings with NotSupportedException;
/// mixing supported encodings in Add or using the wrong Download overload raises ArgumentException.
type ITensorBackend =
    inherit IDisposable
    inherit IAddKernel
    inherit ISwiGluKernel
    inherit IRmsNormKernel
    inherit IEmbeddingKernel
    inherit IRowCopyKernel
    inherit IMatVecKernel
    inherit IMatMulKernel
    inherit IRopeKernel
    inherit IAttentionTemperatureKernel
    inherit IKeyValueCacheKernel
    inherit IAttentionKernel

    abstract Backend: BackendKind

    /// Allocates zero-initialized F32 storage. Raises ArgumentNullException for a null shape
    /// or ObjectDisposedException after backend disposal.
    abstract Allocate: shape: TensorShape -> ITensor

    /// Allocates zero-initialized storage in the requested encoding. CPU and Metal support F32
    /// and F16; other encodings raise NotSupportedException, even for empty tensors.
    /// Shape validation and lifetime rules are the same as for Allocate(shape).
    abstract Allocate: shape: TensorShape * encoding: TensorEncoding -> ITensor

    /// Copies exactly Shape.ElementCount host values, in row-major order, into new storage before
    /// returning; the caller may reuse its buffer immediately. Raises ArgumentNullException for a
    /// null shape, ArgumentException for an incorrect source length, or ObjectDisposedException
    /// after backend disposal.
    abstract Upload: shape: TensorShape * values: ReadOnlyMemory<single> -> ITensor

    /// Copies F16 host values into new F16 storage, preserving their bits, including NaN payloads.
    /// No F32 tensor or conversion buffer is created. Length and lifetime rules match F32 Upload.
    abstract Upload: shape: TensorShape * values: ReadOnlyMemory<Half> -> ITensor

    /// Places a stored matrix on this backend in its encoding, without decoding it. The tensor may
    /// retain the matrix payload instead of copying it, as the CPU backend does, so the tensor
    /// source that produced the matrix must stay alive while the weight tensor is in use.
    /// Raises ArgumentNullException for null, NotSupportedException for an encoding this backend
    /// cannot hold, or ObjectDisposedException after backend disposal.
    abstract UploadWeights: weights: WeightMatrix -> IWeightTensor

    /// Allocates storage for every position of every layer of the layout, about TotalBytes, with no
    /// stored positions. Raises ArgumentNullException for null, NotSupportedException for an
    /// encoding this backend cannot store, or ObjectDisposedException after backend disposal.
    abstract AllocateKeyValueCache: layout: KeyValueCacheLayout -> IKeyValueCache

    /// Waits for previously submitted operations, then copies the tensor to an exact-length host
    /// buffer in row-major order; the buffer is complete on return. Validates before writing.
    /// Raises ArgumentNullException for null, ArgumentException for a foreign tensor or incorrect
    /// destination length or a non-F32 tensor, or ObjectDisposedException for disposed storage or backend.
    abstract Download: tensor: ITensor * destination: Memory<single> -> unit

    /// Copies an F16 tensor to exact-length F16 host storage, preserving bits without conversion.
    /// A non-F16 tensor raises ArgumentException before any write. Ordering, ownership and lifetime
    /// rules match F32 Download; an empty download also waits for earlier submitted operations.
    abstract Download: tensor: ITensor * destination: Memory<Half> -> unit

    /// Waits until every previously submitted operation has completed, and raises any device
    /// failure they reported. Raises ObjectDisposedException after backend disposal.
    abstract Synchronize: unit -> unit
