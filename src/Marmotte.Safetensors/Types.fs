namespace Marmotte.Safetensors

/// Element types named by the dtype field of a tensor descriptor, independent of GGUF's GgmlType.
/// Wire names and widths follow Dtype and Dtype::bitsize in SafeTensors' reference implementation:
/// https://github.com/huggingface/safetensors/blob/main/safetensors/src/tensor.rs (checked 2026-10-24).
[<RequireQualifiedAccess>]
type SafetensorsDtype =
    | Bool
    | F4
    | F6_E2M3
    | F6_E3M2
    | U8
    | I8
    | F8_E5M2
    | F8_E4M3
    | F8_E8M0
    | F8_E4M3FNUZ
    | F8_E5M2FNUZ
    | I16
    | U16
    | F16
    | BF16
    | I32
    | U32
    | F32
    | C64
    | F64
    | I64
    | U64

    /// Name used in the JSON header, such as "BOOL", "BF16" or "F8_E4M3".
    member this.WireName =
        match this with
        | SafetensorsDtype.Bool -> "BOOL"
        | SafetensorsDtype.F4 -> "F4"
        | SafetensorsDtype.F6_E2M3 -> "F6_E2M3"
        | SafetensorsDtype.F6_E3M2 -> "F6_E3M2"
        | SafetensorsDtype.U8 -> "U8"
        | SafetensorsDtype.I8 -> "I8"
        | SafetensorsDtype.F8_E5M2 -> "F8_E5M2"
        | SafetensorsDtype.F8_E4M3 -> "F8_E4M3"
        | SafetensorsDtype.F8_E8M0 -> "F8_E8M0"
        | SafetensorsDtype.F8_E4M3FNUZ -> "F8_E4M3FNUZ"
        | SafetensorsDtype.F8_E5M2FNUZ -> "F8_E5M2FNUZ"
        | SafetensorsDtype.I16 -> "I16"
        | SafetensorsDtype.U16 -> "U16"
        | SafetensorsDtype.F16 -> "F16"
        | SafetensorsDtype.BF16 -> "BF16"
        | SafetensorsDtype.I32 -> "I32"
        | SafetensorsDtype.U32 -> "U32"
        | SafetensorsDtype.F32 -> "F32"
        | SafetensorsDtype.C64 -> "C64"
        | SafetensorsDtype.F64 -> "F64"
        | SafetensorsDtype.I64 -> "I64"
        | SafetensorsDtype.U64 -> "U64"

    /// Bits per element. F4 and the F6 types pack several elements into each byte;
    /// C64 is a complex number made of two 32-bit floating-point parts.
    member this.BitsPerElement : uint64 =
        match this with
        | SafetensorsDtype.F4 -> 4UL
        | SafetensorsDtype.F6_E2M3
        | SafetensorsDtype.F6_E3M2 -> 6UL
        | SafetensorsDtype.Bool
        | SafetensorsDtype.U8
        | SafetensorsDtype.I8
        | SafetensorsDtype.F8_E5M2
        | SafetensorsDtype.F8_E4M3
        | SafetensorsDtype.F8_E8M0
        | SafetensorsDtype.F8_E4M3FNUZ
        | SafetensorsDtype.F8_E5M2FNUZ -> 8UL
        | SafetensorsDtype.I16
        | SafetensorsDtype.U16
        | SafetensorsDtype.F16
        | SafetensorsDtype.BF16 -> 16UL
        | SafetensorsDtype.I32
        | SafetensorsDtype.U32
        | SafetensorsDtype.F32 -> 32UL
        | SafetensorsDtype.C64
        | SafetensorsDtype.F64
        | SafetensorsDtype.I64
        | SafetensorsDtype.U64 -> 64UL

    /// Every dtype, in the reference implementation's declaration order.
    static member internal All =
        [| SafetensorsDtype.Bool
           SafetensorsDtype.F4
           SafetensorsDtype.F6_E2M3
           SafetensorsDtype.F6_E3M2
           SafetensorsDtype.U8
           SafetensorsDtype.I8
           SafetensorsDtype.F8_E5M2
           SafetensorsDtype.F8_E4M3
           SafetensorsDtype.F8_E8M0
           SafetensorsDtype.F8_E4M3FNUZ
           SafetensorsDtype.F8_E5M2FNUZ
           SafetensorsDtype.I16
           SafetensorsDtype.U16
           SafetensorsDtype.F16
           SafetensorsDtype.BF16
           SafetensorsDtype.I32
           SafetensorsDtype.U32
           SafetensorsDtype.F32
           SafetensorsDtype.C64
           SafetensorsDtype.F64
           SafetensorsDtype.I64
           SafetensorsDtype.U64 |]

/// A tensor descriptor from the JSON header; weights are never loaded by the parser.
type SafetensorsTensorInfo =
    { Name: string
      Dtype: SafetensorsDtype
      /// Dimensions in row-major (C) order, with the contiguous dimension last; empty for a scalar.
      Shape: uint64 array
      ElementCount: uint64
      /// Start of the payload relative to the tensor data buffer (the first data_offsets value).
      Offset: uint64
      /// Payload size in bytes, from data_offsets, validated against the shape and dtype.
      ByteSize: uint64
      /// Absolute position of the payload in the source stream.
      AbsoluteOffset: int64 }

/// Parsed Safetensors index, independent of the lifetime of the source stream.
type SafetensorsFile =
    { /// Size of the JSON header in bytes, including its trailing whitespace.
      HeaderLength: uint64
      /// Entries of the optional __metadata__ object; empty when it is absent.
      Metadata: Map<string, string>
      /// Descriptors ordered by data offset, then end offset and name; JSON order is not preserved.
      Tensors: SafetensorsTensorInfo array
      /// Absolute position of the tensor data buffer in the source stream.
      DataOffset: int64
      /// Size of the tensor data buffer in bytes. The tensors cover it exactly, without gaps.
      DataLength: uint64 }

/// Allocation limits for reading untrusted files. Sizes are in bytes.
type SafetensorsReadOptions =
    { MaxHeaderBytes: int
      MaxTensors: int
      MaxRank: int }

    /// The header limit matches the reference implementation's MAX_HEADER_SIZE;
    /// the tensor and rank limits are Marmotte's own.
    static member Default =
        { MaxHeaderBytes = 100_000_000
          MaxTensors = 100_000
          MaxRank = 64 }
