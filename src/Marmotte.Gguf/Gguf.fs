namespace Marmotte.Gguf

open System
open System.Buffers.Binary
open System.Collections.Generic
open System.IO
open System.Text

type private GgufReader(stream: Stream, options: GgufReadOptions) =
    let start = stream.Position
    let length = stream.Length
    let buffer = Array.zeroCreate<byte> 8
    let utf8 = UTF8Encoding(false, true)
    let mutable bigEndian = false

    member _.Start = start
    member _.Length = length
    member _.Position = stream.Position
    member _.BigEndian with set value = bigEndian <- value

    member _.Fail(message: string) : 'T =
        raise (InvalidDataException($"Invalid GGUF at byte {stream.Position}: {message}"))

    member this.EnsureAvailable(count: uint64) =
        if stream.Position > length || count > uint64 (length - stream.Position) then
            this.Fail("Unexpected end of file.")

        if count > uint64 (max 0L (options.MaxHeaderBytes - (stream.Position - start))) then
            this.Fail("Header exceeds MaxHeaderBytes.")

    member this.ReadBytes(count: int) =
        this.EnsureAvailable(uint64 count)
        let bytes = Array.zeroCreate<byte> count
        try stream.ReadExactly(bytes, 0, count)
        with :? EndOfStreamException -> this.Fail("Unexpected end of file.")
        bytes

    member this.Fill(count: int) =
        this.EnsureAvailable(uint64 count)
        try stream.ReadExactly(buffer, 0, count)
        with :? EndOfStreamException -> this.Fail("Unexpected end of file.")

    member this.UInt8() =
        this.Fill(1)
        buffer[0]

    member this.UInt16() =
        this.Fill(2)
        if bigEndian then BinaryPrimitives.ReadUInt16BigEndian(buffer.AsSpan(0, 2))
        else BinaryPrimitives.ReadUInt16LittleEndian(buffer.AsSpan(0, 2))

    member this.UInt32() =
        this.Fill(4)
        if bigEndian then BinaryPrimitives.ReadUInt32BigEndian(buffer.AsSpan(0, 4))
        else BinaryPrimitives.ReadUInt32LittleEndian(buffer.AsSpan(0, 4))

    member this.UInt64() =
        this.Fill(8)
        if bigEndian then BinaryPrimitives.ReadUInt64BigEndian(buffer.AsSpan())
        else BinaryPrimitives.ReadUInt64LittleEndian(buffer.AsSpan())

    member this.Count(label: string, count: uint64, limit: int, minimumBytes: uint64) =
        if count > uint64 limit then
            this.Fail($"{label} exceeds the configured limit ({limit}).")
        this.EnsureAvailable(count * minimumBytes)
        int count

    member this.String(maxBytes: int) =
        let count = this.Count("String length", this.UInt64(), min maxBytes options.MaxStringBytes, 1UL)
        let bytes = this.ReadBytes(count)
        try utf8.GetString(bytes)
        with :? DecoderFallbackException -> this.Fail("Invalid UTF-8 string.")

/// Reads GGUF v2/v3 headers, metadata and tensor descriptors without loading weights.
[<RequireQualifiedAccess>]
module Gguf =
    // (elements per block, bytes per block), matching gguf-py's GGML_QUANT_SIZES.
    let private blockLayout = function
        | GgmlType.F32 | GgmlType.I32 -> Some(1UL, 4UL)
        | GgmlType.F16 | GgmlType.BF16 | GgmlType.I16 -> Some(1UL, 2UL)
        | GgmlType.I8 -> Some(1UL, 1UL)
        | GgmlType.I64 | GgmlType.F64 -> Some(1UL, 8UL)
        | GgmlType.Q4_0 | GgmlType.IQ4_NL -> Some(32UL, 18UL)
        | GgmlType.Q4_1 -> Some(32UL, 20UL)
        | GgmlType.Q5_0 -> Some(32UL, 22UL)
        | GgmlType.Q5_1 -> Some(32UL, 24UL)
        | GgmlType.Q8_0 -> Some(32UL, 34UL)
        | GgmlType.Q8_1 -> Some(32UL, 36UL)
        | GgmlType.Q2_K -> Some(256UL, 84UL)
        | GgmlType.Q3_K -> Some(256UL, 110UL)
        | GgmlType.Q4_K -> Some(256UL, 144UL)
        | GgmlType.Q5_K -> Some(256UL, 176UL)
        | GgmlType.Q6_K -> Some(256UL, 210UL)
        | GgmlType.Q8_K -> Some(256UL, 292UL)
        | GgmlType.IQ2_XXS | GgmlType.TQ2_0 -> Some(256UL, 66UL)
        | GgmlType.IQ2_XS -> Some(256UL, 74UL)
        | GgmlType.IQ3_XXS -> Some(256UL, 98UL)
        | GgmlType.IQ1_S -> Some(256UL, 50UL)
        | GgmlType.IQ3_S -> Some(256UL, 110UL)
        | GgmlType.IQ2_S -> Some(256UL, 82UL)
        | GgmlType.IQ4_XS -> Some(256UL, 136UL)
        | GgmlType.IQ1_M -> Some(256UL, 56UL)
        | GgmlType.TQ1_0 -> Some(256UL, 54UL)
        | GgmlType.MXFP4 -> Some(32UL, 17UL)
        | GgmlType.NVFP4 -> Some(64UL, 36UL)
        | GgmlType.Q1_0 -> Some(128UL, 18UL)
        | GgmlType.Q2_0 -> Some(64UL, 18UL)
        | _ -> None

    let private minimumValueBytes = function
        | GgufValueType.UInt8 | GgufValueType.Int8 | GgufValueType.Bool -> 1UL
        | GgufValueType.UInt16 | GgufValueType.Int16 -> 2UL
        | GgufValueType.UInt32 | GgufValueType.Int32 | GgufValueType.Float32 -> 4UL
        | GgufValueType.Array -> 12UL
        | _ -> 8UL

    let private validateOptions options =
        if options.MaxStringBytes < 0 || options.MaxArrayElements < 0
           || options.MaxMetadataEntries < 0 || options.MaxTensors < 0
           || options.MaxArrayDepth < 0 || options.MaxArrayDepth > 64
           || options.MaxMetadataValues < 0 || options.MaxHeaderBytes < 24L then
            invalidArg (nameof options) "Limits must be nonnegative, MaxArrayDepth at most 64, and MaxHeaderBytes at least 24."

    /// Reads from the current position of a readable, seekable stream and leaves it open
    /// at DataOffset on success. Invalid files raise InvalidDataException with a byte position.
    /// The stream must contain the payload, but the parser only seeks over header padding.
    let readWithOptions (options: GgufReadOptions) (stream: Stream) : GgufFile =
        ArgumentNullException.ThrowIfNull(stream)
        if not stream.CanRead || not stream.CanSeek then
            invalidArg (nameof stream) "The GGUF stream must be readable and seekable."
        validateOptions options
        let reader = GgufReader(stream, options)

        if reader.ReadBytes(4) <> [| 0x47uy; 0x47uy; 0x55uy; 0x46uy |] then
            reader.Fail("Invalid magic; expected GGUF.")

        let rawVersion = reader.UInt32()
        let version, endianness =
            match rawVersion with
            | 2u | 3u -> rawVersion, GgufEndianness.LittleEndian
            | 0x03000000u ->
                reader.BigEndian <- true
                3u, GgufEndianness.BigEndian
            | _ -> reader.Fail($"Unsupported GGUF version ({rawVersion}); expected v2 or v3.")

        let tensorCount = reader.UInt64()
        let metadataCount = reader.UInt64()
        let tensorCount = reader.Count("Tensor count", tensorCount, options.MaxTensors, 24UL)
        let metadataCount = reader.Count("Metadata count", metadataCount, options.MaxMetadataEntries, 13UL)
        let mutable remainingValues = options.MaxMetadataValues

        let readValueType () =
            let tag = reader.UInt32()
            if tag > 12u then reader.Fail($"Unknown metadata value type ({tag}).")
            LanguagePrimitives.EnumOfValue<uint32, GgufValueType> tag

        let rec readValue depth valueType =
            if remainingValues = 0 then reader.Fail("Metadata exceeds MaxMetadataValues.")
            remainingValues <- remainingValues - 1
            match valueType with
            | GgufValueType.UInt8 -> GgufValue.UInt8(reader.UInt8())
            | GgufValueType.Int8 -> GgufValue.Int8(sbyte (reader.UInt8()))
            | GgufValueType.UInt16 -> GgufValue.UInt16(reader.UInt16())
            | GgufValueType.Int16 -> GgufValue.Int16(int16 (reader.UInt16()))
            | GgufValueType.UInt32 -> GgufValue.UInt32(reader.UInt32())
            | GgufValueType.Int32 -> GgufValue.Int32(int32 (reader.UInt32()))
            | GgufValueType.Float32 -> GgufValue.Float32(BitConverter.Int32BitsToSingle(int32 (reader.UInt32())))
            | GgufValueType.Bool ->
                match reader.UInt8() with
                | 0uy -> GgufValue.Bool false
                | 1uy -> GgufValue.Bool true
                | value -> reader.Fail($"Invalid boolean ({value}); expected 0 or 1.")
            | GgufValueType.String -> GgufValue.String(reader.String(options.MaxStringBytes))
            | GgufValueType.UInt64 -> GgufValue.UInt64(reader.UInt64())
            | GgufValueType.Int64 -> GgufValue.Int64(int64 (reader.UInt64()))
            | GgufValueType.Float64 -> GgufValue.Float64(BitConverter.Int64BitsToDouble(int64 (reader.UInt64())))
            | GgufValueType.Array ->
                if depth >= options.MaxArrayDepth then reader.Fail("Array exceeds MaxArrayDepth.")
                let elementType = readValueType ()
                let count = reader.Count("Array length", reader.UInt64(), min options.MaxArrayElements remainingValues, minimumValueBytes elementType)
                GgufValue.Array(elementType, Array.init count (fun _ -> readValue (depth + 1) elementType))
            | _ -> reader.Fail("Unknown metadata value type.")

        let keys = HashSet<string>(StringComparer.Ordinal)
        let metadata =
            Array.init metadataCount (fun _ ->
                let key = reader.String(65535)
                // Numeric segments occur in standardized keys such as general.base_model.0.name.
                if key.Split('.') |> Array.exists (fun segment ->
                    String.IsNullOrEmpty(segment)
                    || segment |> Seq.exists (fun c -> not (c >= 'a' && c <= 'z' || c >= '0' && c <= '9' || c = '_'))) then
                    reader.Fail($"Invalid metadata key '{key}'.")
                if not (keys.Add(key)) then reader.Fail($"Duplicate metadata key '{key}'.")
                key, readValue 0 (readValueType ()))
            |> Map.ofArray

        let alignment =
            match Map.tryFind "general.alignment" metadata with
            | None -> 32u
            | Some(GgufValue.UInt32 value) when value > 0u && value % 8u = 0u -> value
            | _ -> reader.Fail("general.alignment must be a nonzero uint32 multiple of 8.")

        let multiply left right =
            if right <> 0UL && left > UInt64.MaxValue / right then reader.Fail("Tensor size overflows uint64.")
            left * right

        let names = HashSet<string>(StringComparer.Ordinal)
        let tensors =
            Array.init tensorCount (fun _ ->
                let name = reader.String(64)
                if String.IsNullOrEmpty(name) || name.Contains('\000') then reader.Fail("Invalid tensor name.")
                if not (names.Add(name)) then reader.Fail($"Duplicate tensor name '{name}'.")
                let rank = reader.UInt32()
                if rank > 4u then reader.Fail($"Invalid tensor rank ({rank}); expected at most 4.")
                let dimensions = Array.init (int rank) (fun _ -> reader.UInt64())
                let elementCount =
                    if dimensions |> Array.contains 0UL then 0UL
                    else Array.fold multiply 1UL dimensions
                let tensorType = LanguagePrimitives.EnumOfValue<uint32, GgmlType>(reader.UInt32())
                let offset = reader.UInt64()
                if offset % uint64 alignment <> 0UL then reader.Fail($"Tensor '{name}' has an unaligned offset.")
                let byteSize =
                    blockLayout tensorType |> Option.map (fun (blockSize, typeSize) ->
                        let rowSize = if dimensions.Length = 0 then 1UL else dimensions[0]
                        if rowSize % blockSize <> 0UL then reader.Fail($"Tensor '{name}' has an incomplete quantization block.")
                        multiply (elementCount / blockSize) typeSize)
                { Name = name; Dimensions = dimensions; TensorType = tensorType
                  ElementCount = elementCount; Offset = offset; AbsoluteOffset = 0L; ByteSize = byteSize })

        let relativePosition = uint64 (reader.Position - reader.Start)
        let padding = (uint64 alignment - relativePosition % uint64 alignment) % uint64 alignment
        reader.EnsureAvailable(padding)
        let dataOffset = reader.Position + int64 padding
        let dataLength = uint64 (reader.Length - dataOffset)
        let tensors =
            tensors |> Array.map (fun tensor ->
                if tensor.Offset > dataLength || (tensor.Offset = dataLength && tensor.ElementCount <> 0UL) then
                    reader.Fail($"Tensor '{tensor.Name}' starts outside the data section.")
                match tensor.ByteSize with
                | Some size when size > dataLength - tensor.Offset -> reader.Fail($"Tensor '{tensor.Name}' payload is truncated.")
                | _ -> ()
                { tensor with AbsoluteOffset = dataOffset + int64 tensor.Offset })

        // Known payloads may not overlap, even when descriptors are stored out of offset order.
        let sorted = tensors |> Array.filter (fun tensor -> tensor.ElementCount <> 0UL) |> Array.sortBy (fun tensor -> tensor.Offset)
        for index = 1 to sorted.Length - 1 do
            let previous, current = sorted[index - 1], sorted[index]
            if previous.Offset = current.Offset
               || (previous.ByteSize |> Option.exists (fun size -> size > current.Offset - previous.Offset)) then
                reader.Fail($"Tensor '{current.Name}' overlaps another tensor.")

        stream.Position <- dataOffset
        { Version = version; Endianness = endianness; Metadata = metadata
          Tensors = tensors; Alignment = alignment; DataOffset = dataOffset }

    /// Reads with the default allocation limits. The caller retains ownership of the stream.
    let read stream = readWithOptions GgufReadOptions.Default stream

    /// Opens a file, reads its index and closes it without retaining a handle to the weights.
    let readFileWithOptions options path =
        use stream = File.OpenRead(path)
        readWithOptions options stream

    /// Reads a file with the default allocation limits.
    let readFile path = readFileWithOptions GgufReadOptions.Default path
