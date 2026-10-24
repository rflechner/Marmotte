namespace Marmotte.Safetensors

open System
open System.Buffers.Binary
open System.Collections.Generic
open System.IO
open System.Text.Json
open System.Text.Unicode

/// Reads Safetensors headers and tensor descriptors without loading weights.
[<RequireQualifiedAccess>]
module Safetensors =
    let private headerLengthBytes = 8

    // A valid header nests at most three levels: the root object, a descriptor and its shape.
    let private headerDepth = 3

    let private metadataKey = "__metadata__"

    let private descriptorFields = [| "dtype"; "shape"; "data_offsets" |]

    let private failAt (position: int64) (message: string) : 'T =
        raise (InvalidDataException($"Invalid Safetensors file at byte {position}: {message}"))

    let private failHeader (message: string) : 'T =
        raise (InvalidDataException($"Invalid Safetensors header: {message}"))

    let private validateOptions (options: SafetensorsReadOptions) =
        if options.MaxHeaderBytes < 0 || options.MaxTensors < 0 || options.MaxRank < 0 then
            invalidArg (nameof options) "Limits must be nonnegative."

    let private readExactly (stream: Stream) (buffer: byte array) =
        try
            stream.ReadExactly(buffer)
        with :? EndOfStreamException ->
            failAt stream.Position "Unexpected end of file."

    /// Reads the little-endian length prefix and the JSON header, checking the length
    /// against MaxHeaderBytes and the stream length before allocating.
    let private readHeaderBytes (options: SafetensorsReadOptions) (stream: Stream) start length =
        if length - start < int64 headerLengthBytes then
            failAt start "Unexpected end of file; expected an 8-byte header length."

        let prefix = Array.zeroCreate<byte> headerLengthBytes
        readExactly stream prefix
        let headerLength = BinaryPrimitives.ReadUInt64LittleEndian(prefix)

        if headerLength > uint64 options.MaxHeaderBytes then
            failAt start $"Header length ({headerLength}) exceeds MaxHeaderBytes ({options.MaxHeaderBytes})."
        if headerLength > uint64 (length - stream.Position) then
            failAt start $"Header length ({headerLength}) exceeds the remaining file size."

        let header = Array.zeroCreate<byte> (int headerLength)
        readExactly stream header
        header

    // The format requires a leading '{'. Trailing whitespace is left to the JSON parser,
    // which accepts any JSON whitespace, as the reference implementation does.
    let private validateHeaderText (header: byte array) =
        if header.Length = 0 || header[0] <> byte '{' then
            failHeader "The JSON header must start with '{'."
        if not (Utf8.IsValid(ReadOnlySpan<byte>(header))) then
            failHeader "The JSON header is not valid UTF-8."

    let private readUInt64 description (element: JsonElement) =
        let isUInt64, value =
            if element.ValueKind = JsonValueKind.Number then element.TryGetUInt64()
            else false, 0UL

        if not isUInt64 then
            failHeader $"{description} must be a nonnegative integer within the uint64 range."
        value

    let private checkedMultiply name (left: uint64) (right: uint64) =
        if right <> 0UL && left > UInt64.MaxValue / right then
            failHeader $"Tensor '{name}' size overflows uint64."
        left * right

    let private readDtype name (element: JsonElement) =
        let wireName =
            if element.ValueKind = JsonValueKind.String then element.GetString()
            else failHeader $"Tensor '{name}' dtype must be a string."

        match SafetensorsDtype.All |> Array.tryFind (fun dtype -> dtype.WireName = wireName) with
        | Some dtype -> dtype
        | None -> failHeader $"Tensor '{name}' has an unsupported dtype '{wireName}'."

    let private readShape (options: SafetensorsReadOptions) name (element: JsonElement) =
        if element.ValueKind <> JsonValueKind.Array then
            failHeader $"Tensor '{name}' shape must be an array."

        let rank = element.GetArrayLength()
        if rank > options.MaxRank then
            failHeader $"Tensor '{name}' rank ({rank}) exceeds MaxRank ({options.MaxRank})."

        element.EnumerateArray()
        |> Seq.mapi (fun index dimension -> readUInt64 $"Tensor '{name}' dimension {index}" dimension)
        |> Seq.toArray

    /// Returns the inclusive start and exclusive end, relative to the tensor data buffer.
    let private readDataOffsets name (element: JsonElement) =
        if element.ValueKind <> JsonValueKind.Array || element.GetArrayLength() <> 2 then
            failHeader $"Tensor '{name}' data_offsets must be an array of two integers."

        let startOffset = readUInt64 $"Tensor '{name}' data_offsets start" element[0]
        let endOffset = readUInt64 $"Tensor '{name}' data_offsets end" element[1]
        if endOffset < startOffset then
            failHeader $"Tensor '{name}' data_offsets end ({endOffset}) precedes its start ({startOffset})."

        startOffset, endOffset

    /// Payload bytes implied by the shape and dtype. Sub-byte dtypes must fill whole bytes.
    let private payloadSize name (dtype: SafetensorsDtype) elementCount =
        let bitCount = checkedMultiply name elementCount dtype.BitsPerElement
        if bitCount % 8UL <> 0UL then
            failHeader $"Tensor '{name}' does not fill a whole number of bytes ({bitCount} bits of {dtype.WireName})."

        bitCount / 8UL

    /// Returns the dtype, shape and data_offsets fields, rejecting duplicate fields and
    /// fields whose meaning is unknown rather than ignoring them.
    let private readDescriptorFields name (element: JsonElement) =
        if element.ValueKind <> JsonValueKind.Object then
            failHeader $"Tensor '{name}' must be described by a JSON object."

        let fields = Dictionary<string, JsonElement>(StringComparer.Ordinal)
        for field in element.EnumerateObject() do
            let fieldName = field.Name
            if not (Array.contains fieldName descriptorFields) then
                failHeader $"Tensor '{name}' has an unknown field '{fieldName}'."
            if not (fields.TryAdd(fieldName, field.Value)) then
                failHeader $"Tensor '{name}' has a duplicate field '{fieldName}'."

        let required fieldName =
            match fields.TryGetValue fieldName with
            | true, value -> value
            | false, _ -> failHeader $"Tensor '{name}' has no {fieldName} field."

        required "dtype", required "shape", required "data_offsets"

    /// Reads one descriptor. AbsoluteOffset stays 0 until the data buffer is validated.
    let private readDescriptor options name (element: JsonElement) =
        let dtypeElement, shapeElement, dataOffsetsElement = readDescriptorFields name element
        let dtype = readDtype name dtypeElement
        let shape = readShape options name shapeElement
        let startOffset, endOffset = readDataOffsets name dataOffsetsElement

        let elementCount =
            if shape |> Array.contains 0UL then 0UL
            else Array.fold (checkedMultiply name) 1UL shape

        let byteSize = payloadSize name dtype elementCount
        if endOffset - startOffset <> byteSize then
            failHeader $"Tensor '{name}' data_offsets cover {endOffset - startOffset} bytes; its shape and dtype require {byteSize}."

        { Name = name
          Dtype = dtype
          Shape = shape
          ElementCount = elementCount
          Offset = startOffset
          ByteSize = byteSize
          AbsoluteOffset = 0L }

    let private readMetadata (element: JsonElement) =
        if element.ValueKind <> JsonValueKind.Object then
            failHeader "__metadata__ must be a JSON object of strings."

        let keys = HashSet<string>(StringComparer.Ordinal)

        let readEntry (property: JsonProperty) =
            let key = property.Name
            if not (keys.Add(key)) then
                failHeader $"Duplicate __metadata__ key '{key}'."
            if property.Value.ValueKind <> JsonValueKind.String then
                failHeader $"__metadata__ value for '{key}' must be a string."

            key, property.Value.GetString()

        element.EnumerateObject() |> Seq.map readEntry |> Map.ofSeq

    /// Interprets the root object. Keys are compared after JSON unescaping, so duplicates
    /// are rejected even when escaped differently, instead of keeping the last value.
    let private readEntries (options: SafetensorsReadOptions) (root: JsonElement) =
        let keys = HashSet<string>(StringComparer.Ordinal)
        let descriptors = ResizeArray<SafetensorsTensorInfo>()
        let mutable metadata = Map.empty

        for property in root.EnumerateObject() do
            let key = property.Name
            if not (keys.Add(key)) then
                failHeader $"Duplicate key '{key}'."

            if key = metadataKey then
                metadata <- readMetadata property.Value
            else
                if descriptors.Count = options.MaxTensors then
                    failHeader $"Tensor count exceeds MaxTensors ({options.MaxTensors})."
                descriptors.Add(readDescriptor options key property.Value)

        metadata, descriptors.ToArray()

    let private parseHeader options (header: byte array) =
        validateHeaderText header
        let documentOptions = JsonDocumentOptions(MaxDepth = headerDepth)

        try
            use document = JsonDocument.Parse(ReadOnlyMemory<byte>(header), documentOptions)
            readEntries options document.RootElement
        with
        | :? JsonException as error -> failHeader error.Message
        // System.Text.Json reports escaped strings with invalid surrogates when they are read.
        | :? InvalidOperationException as error -> failHeader error.Message

    /// Sorts descriptors by range and requires them to cover the data buffer exactly:
    /// each range starts where the previous one ends, and the last one ends with the stream.
    let private validateCoverage dataLength (descriptors: SafetensorsTensorInfo array) =
        let sorted =
            descriptors
            |> Array.sortBy (fun tensor -> tensor.Offset, tensor.Offset + tensor.ByteSize, tensor.Name)

        let mutable cursor = 0UL
        for tensor in sorted do
            if tensor.Offset < cursor then
                failHeader $"Tensor '{tensor.Name}' overlaps the previous tensor."
            if tensor.Offset > cursor then
                failHeader $"{tensor.Offset - cursor} bytes before tensor '{tensor.Name}' belong to no tensor."
            cursor <- tensor.Offset + tensor.ByteSize

        if cursor < dataLength then
            failHeader $"{dataLength - cursor} bytes at the end of the tensor data belong to no tensor."
        if cursor > dataLength then
            failHeader $"Tensor data ends at offset {cursor}, beyond the {dataLength} bytes remaining in the file."

        sorted

    /// Reads from the current position of a readable, seekable stream and leaves it open at
    /// DataOffset on success. The stream must end exactly where the tensor data ends.
    /// Invalid files raise InvalidDataException; I/O errors propagate.
    let readWithOptions (options: SafetensorsReadOptions) (stream: Stream) : SafetensorsFile =
        ArgumentNullException.ThrowIfNull(stream)
        if not stream.CanRead || not stream.CanSeek then
            invalidArg (nameof stream) "The Safetensors stream must be readable and seekable."
        validateOptions options

        let length = stream.Length
        let header = readHeaderBytes options stream stream.Position length
        let metadata, descriptors = parseHeader options header

        let dataOffset = stream.Position
        let dataLength = uint64 (length - dataOffset)
        let tensors =
            validateCoverage dataLength descriptors
            |> Array.map (fun tensor -> { tensor with AbsoluteOffset = dataOffset + int64 tensor.Offset })

        { HeaderLength = uint64 header.Length
          Metadata = metadata
          Tensors = tensors
          DataOffset = dataOffset
          DataLength = dataLength }

    /// Reads with the default allocation limits. The caller retains ownership of the stream.
    let read stream = readWithOptions SafetensorsReadOptions.Default stream

    /// Opens a file, reads its index and closes it without retaining a handle to the weights.
    let readFileWithOptions options path =
        use stream = File.OpenRead(path)
        readWithOptions options stream

    /// Reads a file with the default allocation limits.
    let readFile path = readFileWithOptions SafetensorsReadOptions.Default path
