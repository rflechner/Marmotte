namespace Marmotte.Gguf

/// Byte order used by metadata and tensor data. The magic is always ASCII GGUF.
[<RequireQualifiedAccess>]
type GgufEndianness =
    | LittleEndian
    | BigEndian

/// On-disk tags for GGUF metadata values.
type GgufValueType =
    | UInt8 = 0u
    | Int8 = 1u
    | UInt16 = 2u
    | Int16 = 3u
    | UInt32 = 4u
    | Int32 = 5u
    | Float32 = 6u
    | Bool = 7u
    | String = 8u
    | Array = 9u
    | UInt64 = 10u
    | Int64 = 11u
    | Float64 = 12u

/// Metadata values retain their exact type, including the element type of empty arrays.
[<RequireQualifiedAccess>]
type GgufValue =
    | UInt8 of byte
    | Int8 of sbyte
    | UInt16 of uint16
    | Int16 of int16
    | UInt32 of uint32
    | Int32 of int32
    | Float32 of single
    | Bool of bool
    | String of string
    | Array of elementType: GgufValueType * values: GgufValue array
    | UInt64 of uint64
    | Int64 of int64
    | Float64 of double

/// GGML tensor encodings, distinct from the file-level GgufFileType profiles.
/// Unrecognized numeric values are preserved for future formats.
type GgmlType =
    | F32 = 0u
    | F16 = 1u
    | Q4_0 = 2u
    | Q4_1 = 3u
    | Q5_0 = 6u
    | Q5_1 = 7u
    | Q8_0 = 8u
    | Q8_1 = 9u
    | Q2_K = 10u
    | Q3_K = 11u
    | Q4_K = 12u
    | Q5_K = 13u
    | Q6_K = 14u
    | Q8_K = 15u
    | IQ2_XXS = 16u
    | IQ2_XS = 17u
    | IQ3_XXS = 18u
    | IQ1_S = 19u
    | IQ4_NL = 20u
    | IQ3_S = 21u
    | IQ2_S = 22u
    | IQ4_XS = 23u
    | I8 = 24u
    | I16 = 25u
    | I32 = 26u
    | I64 = 27u
    | F64 = 28u
    | IQ1_M = 29u
    | BF16 = 30u
    | TQ1_0 = 34u
    | TQ2_0 = 35u
    | MXFP4 = 39u
    | NVFP4 = 40u
    | Q1_0 = 41u
    | Q2_0 = 42u

/// File-level quantization profiles stored in general.file_type (llama.cpp's llama_ftype).
/// Except F32, these describe the predominant encoding or a mixture of tensor encodings.
/// Their numeric IDs are independent of GgmlType. Unrecognized IDs are preserved.
type GgufFileType =
    | F32 = 0u
    | F16 = 1u
    | Q4_0 = 2u
    | Q4_1 = 3u
    | Q8_0 = 7u
    | Q5_0 = 8u
    | Q5_1 = 9u
    | Q2_K = 10u
    | Q3_K_S = 11u
    | Q3_K_M = 12u
    | Q3_K_L = 13u
    | Q4_K_S = 14u
    | Q4_K_M = 15u
    | Q5_K_S = 16u
    | Q5_K_M = 17u
    | Q6_K = 18u
    | IQ2_XXS = 19u
    | IQ2_XS = 20u
    | Q2_K_S = 21u
    | IQ3_XS = 22u
    | IQ3_XXS = 23u
    | IQ1_S = 24u
    | IQ4_NL = 25u
    | IQ3_S = 26u
    | IQ3_M = 27u
    | IQ2_S = 28u
    | IQ2_M = 29u
    | IQ4_XS = 30u
    | IQ1_M = 31u
    | BF16 = 32u
    | TQ1_0 = 36u
    | TQ2_0 = 37u
    | MXFP4_MOE = 38u
    | NVFP4 = 39u
    | Q1_0 = 40u
    | Q2_0 = 41u

/// A tensor descriptor; weights are never loaded by the parser.
type GgufTensorInfo =
    { Name: string
      /// Dimensions in GGML order, with the contiguous dimension first; empty for a scalar.
      Dimensions: uint64 array
      TensorType: GgmlType
      ElementCount: uint64
      /// Offset relative to the tensor data section.
      Offset: uint64
      /// Absolute position in the source stream.
      AbsoluteOffset: int64
      /// None for unrecognized encodings, whose payload size cannot be validated.
      ByteSize: uint64 option }

/// Parsed GGUF index, independent of the lifetime of the source stream.
type GgufFile =
    { Version: uint32
      Endianness: GgufEndianness
      Metadata: Map<string, GgufValue>
      Tensors: GgufTensorInfo array
      Alignment: uint32
      /// Absolute position of the aligned tensor data section in the source stream.
      DataOffset: int64 }

    /// File-level profile, when general.file_type is a uint32; otherwise None.
    /// Never inferred from tensor encodings. The original metadata remains available.
    member this.FileType =
        match Map.tryFind "general.file_type" this.Metadata with
        | Some(GgufValue.UInt32 value) ->
            Some(LanguagePrimitives.EnumOfValue<uint32, GgufFileType> value)
        | _ -> None

/// Allocation and nesting limits for reading untrusted files. Sizes are in bytes.
type GgufReadOptions =
    { MaxStringBytes: int
      MaxArrayElements: int
      MaxMetadataEntries: int
      MaxTensors: int
      MaxArrayDepth: int
      MaxMetadataValues: int
      MaxHeaderBytes: int64 }

    static member Default =
        { MaxStringBytes = 16 * 1024 * 1024
          MaxArrayElements = 1_000_000
          MaxMetadataEntries = 100_000
          MaxTensors = 100_000
          MaxArrayDepth = 16
          MaxMetadataValues = 2_000_000
          MaxHeaderBytes = 256L * 1024L * 1024L }

/// Output encodings implemented by the native quantizer, not the full set of GGUF profiles.
[<RequireQualifiedAccess>]
type GgufQuantizationFormat =
    | Q4_0
    | Q8_0

/// File sizes are in bytes. Preserved tensors retain their original floating-point encoding.
type GgufQuantizationResult =
    { Format: GgufQuantizationFormat
      QuantizedTensors: int
      PreservedTensors: int
      InputBytes: int64
      OutputBytes: int64 }
