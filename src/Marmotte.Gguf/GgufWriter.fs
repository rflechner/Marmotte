namespace Marmotte.Gguf

open System
open System.IO
open System.Text

/// Writes validated, parsed metadata and planned descriptors as GGUF v3 little-endian.
module internal GgufWriter =
    let align (position: int64) (alignment: uint32) =
        let alignment = int64 alignment
        let padding = (alignment - position % alignment) % alignment
        if position > Int64.MaxValue - padding then
            raise (InvalidDataException("The aligned output position exceeds the file size limit."))
        position + padding

    let private writeString (writer: BinaryWriter) (value: string) =
        let bytes = Encoding.UTF8.GetBytes value
        writer.Write(uint64 bytes.Length)
        writer.Write(bytes)

    let private valueType = function
        | GgufValue.UInt8 _ -> GgufValueType.UInt8
        | GgufValue.Int8 _ -> GgufValueType.Int8
        | GgufValue.UInt16 _ -> GgufValueType.UInt16
        | GgufValue.Int16 _ -> GgufValueType.Int16
        | GgufValue.UInt32 _ -> GgufValueType.UInt32
        | GgufValue.Int32 _ -> GgufValueType.Int32
        | GgufValue.Float32 _ -> GgufValueType.Float32
        | GgufValue.Bool _ -> GgufValueType.Bool
        | GgufValue.String _ -> GgufValueType.String
        | GgufValue.Array _ -> GgufValueType.Array
        | GgufValue.UInt64 _ -> GgufValueType.UInt64
        | GgufValue.Int64 _ -> GgufValueType.Int64
        | GgufValue.Float64 _ -> GgufValueType.Float64

    let rec private writeValue (writer: BinaryWriter) = function
        | GgufValue.UInt8 value -> writer.Write(value)
        | GgufValue.Int8 value -> writer.Write(value)
        | GgufValue.UInt16 value -> writer.Write(value)
        | GgufValue.Int16 value -> writer.Write(value)
        | GgufValue.UInt32 value -> writer.Write(value)
        | GgufValue.Int32 value -> writer.Write(value)
        | GgufValue.Float32 value -> writer.Write(value)
        | GgufValue.Bool value -> writer.Write(value)
        | GgufValue.String value -> writeString writer value
        | GgufValue.UInt64 value -> writer.Write(value)
        | GgufValue.Int64 value -> writer.Write(value)
        | GgufValue.Float64 value -> writer.Write(value)
        | GgufValue.Array(elementType, values) ->
            writer.Write(uint32 elementType)
            writer.Write(uint64 values.Length)
            for value in values do writeValue writer value

    // Only used on a new, empty output file. Seeking and extending creates zero-filled padding.
    let writeIndex (stream: Stream) alignment (metadata: Map<string, GgufValue>) (tensors: GgufTensorInfo array) =
        use writer = new BinaryWriter(stream, Encoding.UTF8, true)
        writer.Write(Encoding.ASCII.GetBytes "GGUF")
        writer.Write(3u)
        writer.Write(uint64 tensors.Length)
        writer.Write(uint64 metadata.Count)
        for KeyValue(key, value) in metadata do
            writeString writer key
            writer.Write(uint32 (valueType value))
            writeValue writer value
        for tensor in tensors do
            writeString writer tensor.Name
            writer.Write(uint32 tensor.Dimensions.Length)
            for dimension in tensor.Dimensions do writer.Write(dimension)
            writer.Write(uint32 tensor.TensorType)
            writer.Write(tensor.Offset)
        writer.Flush()
        let dataOffset = align stream.Position alignment
        stream.SetLength(dataOffset)
        stream.Position <- dataOffset
        dataOffset
