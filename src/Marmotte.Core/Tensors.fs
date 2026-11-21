namespace Marmotte.Core

open System
open System.Collections.Generic

/// Tensor encodings Marmotte represents at runtime, independent of GGUF's GgmlType and
/// Safetensors' dtypes. Deliberately narrower than both formats: the encodings of the first
/// target models, the K-quants of the Q2_K to Q6_K file profiles, and Q4_1, Q5_0, Q5_1 and IQ4_NL,
/// which offer key-value cache sizes between Q4_0 and Q8_0. RowDecoder decodes rows of every
/// encoding into F32 for embedding lookup and matrix products; Add, SwiGLU and RMS normalization
/// operate on F32 execution tensors.
/// Block layouts follow the reference block definitions cited in docs/gguf.md and match the
/// GGUF reader's payload size table.
[<RequireQualifiedAccess>]
type TensorEncoding =
    | F32
    | F16
    | BF16
    | Q4_0
    | Q4_1
    | Q5_0
    | Q5_1
    | Q8_0
    | IQ4_NL
    | Q2_K
    | Q3_K
    | Q4_K
    | Q5_K
    | Q6_K

    /// Elements per quantization block; 1 for plain floating-point encodings.
    /// A block never spans two rows, so a row length must be a multiple of this value.
    member this.BlockElements =
        match this with
        | TensorEncoding.F32
        | TensorEncoding.F16
        | TensorEncoding.BF16 -> 1
        | TensorEncoding.Q4_0
        | TensorEncoding.Q4_1
        | TensorEncoding.Q5_0
        | TensorEncoding.Q5_1
        | TensorEncoding.Q8_0
        | TensorEncoding.IQ4_NL -> 32
        | TensorEncoding.Q2_K
        | TensorEncoding.Q3_K
        | TensorEncoding.Q4_K
        | TensorEncoding.Q5_K
        | TensorEncoding.Q6_K -> 256

    /// Bytes per quantization block, including its scales.
    member this.BlockBytes =
        match this with
        | TensorEncoding.F32 -> 4
        | TensorEncoding.F16
        | TensorEncoding.BF16 -> 2
        | TensorEncoding.Q4_0 -> 18
        | TensorEncoding.Q4_1 -> 20
        | TensorEncoding.Q5_0 -> 22
        | TensorEncoding.Q5_1 -> 24
        | TensorEncoding.Q8_0 -> 34
        | TensorEncoding.IQ4_NL -> 18
        | TensorEncoding.Q2_K -> 84
        | TensorEncoding.Q3_K -> 110
        | TensorEncoding.Q4_K -> 144
        | TensorEncoding.Q5_K -> 176
        | TensorEncoding.Q6_K -> 210

/// Format-neutral description of one stored tensor, independent of its payload.
type TensorDescriptor =
    { Name: string
      /// Runtime encoding, or None when Marmotte does not represent the source encoding.
      Encoding: TensorEncoding option
      /// Encoding name in the source format, such as "Q4_K", "BF16" or "F8_E4M3", for diagnostics.
      SourceEncoding: string
      /// Dimensions in row-major order, with the contiguous dimension last; empty for a scalar.
      /// GGUF sources reverse their dimensions into this order without moving any bytes.
      Shape: uint64 array
      ElementCount: uint64
      /// Payload size in bytes; None when the source format cannot determine it.
      ByteSize: uint64 option }

/// Read-only access to the tensors of one stored model, whatever its file format.
/// The source owns its storage: payloads stay valid until it is disposed, and reading
/// a payload after disposal raises ObjectDisposedException.
type ITensorSource =
    inherit IDisposable

    /// Descriptors in the source's order: index order for GGUF, data offset order for Safetensors.
    abstract Tensors: IReadOnlyList<TensorDescriptor>

    /// Finds a descriptor by its exact, case-sensitive name.
    abstract TryFindTensor: name: string -> TensorDescriptor option

    /// Little-endian payload bytes in the descriptor's encoding, with rows stored one after another.
    /// Raises KeyNotFoundException for an unknown name and InvalidDataException when the tensor
    /// has no runtime encoding or Marmotte cannot hold its payload. I/O errors propagate.
    abstract ReadPayload: name: string -> ReadOnlyMemory<byte>
