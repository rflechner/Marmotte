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
    // (elements per block, bytes per block), matching the reference size table cited in docs/gguf.md.
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

    // Minimum encoded sizes, used to reject impossible counts before allocating.
    // A metadata entry has a key length, a value tag and at least one value byte;
    // a tensor descriptor has a name length, a rank, an encoding and an offset.
    let private minimumMetadataEntryBytes = 13UL
    let private minimumTensorDescriptorBytes = 24UL

    let private maxMetadataKeyBytes = 65535
    let private maxTensorNameBytes = 64
    let private maxTensorRank = 4u
    let private defaultAlignment = 32u

    let private validateOptions options =
        let invalidLimits =
            options.MaxStringBytes < 0
            || options.MaxArrayElements < 0
            || options.MaxMetadataEntries < 0
            || options.MaxTensors < 0
            || options.MaxArrayDepth < 0
            || options.MaxArrayDepth > 64
            || options.MaxMetadataValues < 0
            || options.MaxHeaderBytes < 24L

        if invalidLimits then
            invalidArg (nameof options) "Limits must be nonnegative, MaxArrayDepth at most 64, and MaxHeaderBytes at least 24."

    /// Validates the magic and version, then selects the byte order of the remaining fields.
    let private readHeader (reader: GgufReader) =
        if reader.ReadBytes(4) <> [| 0x47uy; 0x47uy; 0x55uy; 0x46uy |] then
            reader.Fail("Invalid magic; expected GGUF.")

        let rawVersion = reader.UInt32()

        match rawVersion with
        | 2u | 3u -> rawVersion, GgufEndianness.LittleEndian
        | 0x03000000u ->
            reader.BigEndian <- true
            3u, GgufEndianness.BigEndian
        | _ -> reader.Fail($"Unsupported GGUF version ({rawVersion}); expected v2 or v3.")

    // Numeric segments occur in standardized keys such as general.base_model.0.name.
    let private isValidKeySegment (segment: string) =
        let isAllowed character =
            (character >= 'a' && character <= 'z')
            || (character >= '0' && character <= '9')
            || character = '_'

        not (String.IsNullOrEmpty segment) && Seq.forall isAllowed segment

    let private isValidMetadataKey (key: string) =
        key.Split('.') |> Array.forall isValidKeySegment

    /// Reads typed metadata entries. MaxMetadataValues is a budget shared by every entry,
    /// including nested array elements, so it bounds the total number of decoded values.
    let private readMetadata (options: GgufReadOptions) (reader: GgufReader) entryCount =
        let mutable remainingValues = options.MaxMetadataValues

        let readValueType () =
            let tag = reader.UInt32()
            if tag > 12u then
                reader.Fail($"Unknown metadata value type ({tag}).")
            LanguagePrimitives.EnumOfValue<uint32, GgufValueType> tag

        let rec readValue depth valueType =
            if remainingValues = 0 then
                reader.Fail("Metadata exceeds MaxMetadataValues.")
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
                if depth >= options.MaxArrayDepth then
                    reader.Fail("Array exceeds MaxArrayDepth.")

                let elementType = readValueType ()
                let rawLength = reader.UInt64()
                let lengthLimit = min options.MaxArrayElements remainingValues
                let length = reader.Count("Array length", rawLength, lengthLimit, minimumValueBytes elementType)
                let values = Array.init length (fun _ -> readValue (depth + 1) elementType)
                GgufValue.Array(elementType, values)
            | _ -> reader.Fail("Unknown metadata value type.")

        let keys = HashSet<string>(StringComparer.Ordinal)

        let readEntry _ =
            let key = reader.String(maxMetadataKeyBytes)
            if not (isValidMetadataKey key) then
                reader.Fail($"Invalid metadata key '{key}'.")
            if not (keys.Add(key)) then
                reader.Fail($"Duplicate metadata key '{key}'.")

            key, readValue 0 (readValueType ())

        Array.init entryCount readEntry |> Map.ofArray

    let private resolveAlignment (reader: GgufReader) (metadata: Map<string, GgufValue>) =
        match Map.tryFind "general.alignment" metadata with
        | None -> defaultAlignment
        | Some(GgufValue.UInt32 value) when value > 0u && value % 8u = 0u -> value
        | _ -> reader.Fail("general.alignment must be a nonzero uint32 multiple of 8.")

    let private checkedMultiply (reader: GgufReader) left right =
        if right <> 0UL && left > UInt64.MaxValue / right then
            reader.Fail("Tensor size overflows uint64.")
        left * right

    /// Payload size for a known encoding, or None when the encoding is not recognized.
    /// Each row, along the contiguous first dimension, must contain whole quantization blocks.
    let private payloadSize (reader: GgufReader) name (dimensions: uint64 array) elementCount tensorType =
        blockLayout tensorType
        |> Option.map (fun (blockSize, typeSize) ->
            let rowSize = if dimensions.Length = 0 then 1UL else dimensions[0]
            if rowSize % blockSize <> 0UL then
                reader.Fail($"Tensor '{name}' has an incomplete quantization block.")
            checkedMultiply reader (elementCount / blockSize) typeSize)

    /// Reads one descriptor. AbsoluteOffset stays 0 until the data section is located.
    let private readTensorDescriptor (reader: GgufReader) alignment (names: HashSet<string>) =
        let name = reader.String(maxTensorNameBytes)
        if String.IsNullOrEmpty(name) || name.Contains('\000') then
            reader.Fail("Invalid tensor name.")
        if not (names.Add(name)) then
            reader.Fail($"Duplicate tensor name '{name}'.")

        let rank = reader.UInt32()
        if rank > maxTensorRank then
            reader.Fail($"Invalid tensor rank ({rank}); expected at most {maxTensorRank}.")

        let dimensions = Array.init (int rank) (fun _ -> reader.UInt64())
        let elementCount =
            if dimensions |> Array.contains 0UL then 0UL
            else Array.fold (checkedMultiply reader) 1UL dimensions

        let tensorType = LanguagePrimitives.EnumOfValue<uint32, GgmlType>(reader.UInt32())
        let offset = reader.UInt64()
        if offset % uint64 alignment <> 0UL then
            reader.Fail($"Tensor '{name}' has an unaligned offset.")

        { Name = name
          Dimensions = dimensions
          TensorType = tensorType
          ElementCount = elementCount
          Offset = offset
          AbsoluteOffset = 0L
          ByteSize = payloadSize reader name dimensions elementCount tensorType }

    let private readTensorDescriptors (reader: GgufReader) alignment tensorCount =
        let names = HashSet<string>(StringComparer.Ordinal)
        Array.init tensorCount (fun _ -> readTensorDescriptor reader alignment names)

    /// Locates the aligned data section after the descriptors, checks each known payload
    /// range against the stream length and resolves absolute tensor positions.
    let private resolveDataSection (reader: GgufReader) alignment (tensors: GgufTensorInfo array) =
        let relativePosition = uint64 (reader.Position - reader.Start)
        let padding = (uint64 alignment - relativePosition % uint64 alignment) % uint64 alignment
        reader.EnsureAvailable(padding)

        let dataOffset = reader.Position + int64 padding
        let dataLength = uint64 (reader.Length - dataOffset)

        let resolve (tensor: GgufTensorInfo) =
            let startsOutside =
                tensor.Offset > dataLength
                || (tensor.Offset = dataLength && tensor.ElementCount <> 0UL)
            if startsOutside then
                reader.Fail($"Tensor '{tensor.Name}' starts outside the data section.")

            match tensor.ByteSize with
            | Some size when size > dataLength - tensor.Offset ->
                reader.Fail($"Tensor '{tensor.Name}' payload is truncated.")
            | _ -> ()

            { tensor with AbsoluteOffset = dataOffset + int64 tensor.Offset }

        dataOffset, Array.map resolve tensors

    // Known payloads may not overlap, even when descriptors are stored out of offset order.
    // Empty tensors are zero-length ranges and may share an offset with another tensor.
    let private validateNoOverlap (reader: GgufReader) (tensors: GgufTensorInfo array) =
        let sorted =
            tensors
            |> Array.filter (fun tensor -> tensor.ElementCount <> 0UL)
            |> Array.sortBy (fun tensor -> tensor.Offset)

        for index = 1 to sorted.Length - 1 do
            let previous = sorted[index - 1]
            let current = sorted[index]
            let gap = current.Offset - previous.Offset
            let previousExtendsIntoCurrent = previous.ByteSize |> Option.exists (fun size -> size > gap)

            if previous.Offset = current.Offset || previousExtendsIntoCurrent then
                reader.Fail($"Tensor '{current.Name}' overlaps another tensor.")

    /// Reads from the current position of a readable, seekable stream and leaves it open
    /// at DataOffset on success. Invalid files raise InvalidDataException with a byte position.
    /// The stream must contain the payload, but the parser only seeks over header padding.
    let readWithOptions (options: GgufReadOptions) (stream: Stream) : GgufFile =
        ArgumentNullException.ThrowIfNull(stream)
        if not stream.CanRead || not stream.CanSeek then
            invalidArg (nameof stream) "The GGUF stream must be readable and seekable."
        validateOptions options

        let reader = GgufReader(stream, options)
        let version, endianness = readHeader reader

        // Both counts belong to the fixed 24-byte header and are read before either is validated.
        let rawTensorCount = reader.UInt64()
        let rawMetadataCount = reader.UInt64()
        let tensorCount = reader.Count("Tensor count", rawTensorCount, options.MaxTensors, minimumTensorDescriptorBytes)
        let metadataCount = reader.Count("Metadata count", rawMetadataCount, options.MaxMetadataEntries, minimumMetadataEntryBytes)

        let metadata = readMetadata options reader metadataCount
        let alignment = resolveAlignment reader metadata
        let descriptors = readTensorDescriptors reader alignment tensorCount

        let dataOffset, tensors = resolveDataSection reader alignment descriptors
        validateNoOverlap reader tensors
        stream.Position <- dataOffset

        { Version = version
          Endianness = endianness
          Metadata = metadata
          Tensors = tensors
          Alignment = alignment
          DataOffset = dataOffset }

    /// Reads with the default allocation limits. The caller retains ownership of the stream.
    let read stream = readWithOptions GgufReadOptions.Default stream

    /// Opens a file, reads its index and closes it without retaining a handle to the weights.
    let readFileWithOptions options path =
        use stream = File.OpenRead(path)
        readWithOptions options stream

    /// Reads a file with the default allocation limits.
    let readFile path = readFileWithOptions GgufReadOptions.Default path
