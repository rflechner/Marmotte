namespace Marmotte.Core

open System

/// Dimensions and storage encodings of a key-value cache. For each of Layers attention blocks,
/// the cache holds the keys, after RoPE, and the values of up to Capacity positions, each a row of
/// KeyValueHeads × HeadDimension values. Attention later reads these rows instead of recomputing
/// the projections of every earlier token.
///
/// Keys and values have separate encodings, because keys usually tolerate quantization less well.
/// F16 halves an F32 cache. Per block of 32 values, Q8_0 stores 34 bytes, Q5_1 24, Q5_0 22, Q4_1
/// 20, and Q4_0 and IQ4_NL 18, at the cost of precision. Every position is a row of whole blocks
/// of its own, for values as for keys, so quantizing the value cache does not depend on how
/// attention is computed.
///
/// Encoding is lossy except in F32: see RowEncoder for rounding and non-finite values.
/// Quantization blocks never span two heads, so that attention can decode one head at a time.
[<Sealed>]
type KeyValueCacheLayout private (layers: int, capacity: int, keyValueHeads: int, headDimension: int,
                                  keyEncoding: TensorEncoding, valueEncoding: TensorEncoding) =
    let rowValues = keyValueHeads * headDimension
    let rowBytes (encoding: TensorEncoding) = rowValues / encoding.BlockElements * encoding.BlockBytes
    let keyRowBytes = rowBytes keyEncoding
    let valueRowBytes = rowBytes valueEncoding

    /// Attention blocks with a cache each: block_count, 26 for the target model.
    member _.Layers = layers

    /// Positions each layer can hold, from 0: the context length the cache allows.
    member _.Capacity = capacity

    /// Key-value heads of each position, shared by groups of query heads: 8 for the target model.
    member _.KeyValueHeads = keyValueHeads

    /// Values per head: 128 for the target model.
    member _.HeadDimension = headDimension

    member _.KeyEncoding = keyEncoding
    member _.ValueEncoding = valueEncoding

    /// Values of one position in one layer, for keys and for values: KeyValueHeads × HeadDimension.
    member _.RowValues = rowValues

    /// Encoded bytes of the keys of one position in one layer.
    member _.KeyRowBytes = keyRowBytes

    /// Encoded bytes of the values of one position in one layer.
    member _.ValueRowBytes = valueRowBytes

    /// Encoded bytes of one position over every layer, keys and values together.
    member _.BytesPerPosition = int64 layers * int64 (keyRowBytes + valueRowBytes)

    /// Encoded bytes of the whole cache at full capacity, which backends may allocate up front.
    member this.TotalBytes = this.BytesPerPosition * int64 capacity

    /// Raises ArgumentOutOfRangeException for a dimension that is not positive, NotSupportedException
    /// for an encoding that RowEncoder cannot write, and ArgumentException for a head dimension that
    /// is not a multiple of an encoding's block (32 values for every quantized encoding it accepts),
    /// or for a layer's keys or values beyond the managed array limit that execution tensors
    /// currently share.
    static member Create(layers: int, capacity: int, keyValueHeads: int, headDimension: int,
                         keyEncoding: TensorEncoding, valueEncoding: TensorEncoding) =
        let requirePositive (name: string) (value: int) =
            if value <= 0 then
                raise (ArgumentOutOfRangeException(name, box value, "Key-value cache dimensions must be positive."))

        requirePositive (nameof layers) layers
        requirePositive (nameof capacity) capacity
        requirePositive (nameof keyValueHeads) keyValueHeads
        requirePositive (nameof headDimension) headDimension

        let requireEncoding (name: string) (encoding: TensorEncoding) =
            if not (RowEncoder.canEncode encoding) then
                raise (NotSupportedException($"A key-value cache cannot store {encoding}, which RowEncoder cannot write."))
            if headDimension % encoding.BlockElements <> 0 then
                invalidArg name $"Heads of {headDimension} values do not hold whole {encoding} blocks of {encoding.BlockElements}."

        requireEncoding (nameof keyEncoding) keyEncoding
        requireEncoding (nameof valueEncoding) valueEncoding

        if int64 keyValueHeads * int64 headDimension > int64 Array.MaxLength then
            invalidArg (nameof headDimension) "A key-value cache row exceeds the managed array limit."

        let rowValues = keyValueHeads * headDimension
        for encoding in [ keyEncoding; valueEncoding ] do
            let rowBytes = int64 (rowValues / encoding.BlockElements) * int64 encoding.BlockBytes
            if int64 capacity * rowBytes > int64 Array.MaxLength then
                invalidArg (nameof capacity) $"{capacity} positions of {encoding} rows exceed the managed array limit of one layer."

        KeyValueCacheLayout(layers, capacity, keyValueHeads, headDimension, keyEncoding, valueEncoding)
