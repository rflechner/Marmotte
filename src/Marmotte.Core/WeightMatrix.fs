namespace Marmotte.Core

open System
open System.Collections.Generic
open System.IO

/// A rank-2 tensor kept in its stored encoding and accessed row by row.
/// Columns is the contiguous dimension: the last row-major dimension, which is the first
/// GGUF dimension. Each row holds whole quantization blocks, so rows can be decoded
/// independently. The payload stays valid as long as the source it was read from.
[<Sealed>]
type WeightMatrix private (name: string, encoding: TensorEncoding, rows: int, columns: int,
                           rowBytes: int, payload: ReadOnlyMemory<byte>) =

    static let invalid (name: string) (message: string) : exn =
        InvalidDataException($"Tensor '{name}' {message}")

    static let dimension (name: string) (label: string) (value: uint64) =
        if value = 0UL || value > uint64 Int32.MaxValue then
            raise (invalid name $"has {value} {label}; a weight matrix needs 1 to {Int32.MaxValue} {label}.")
        int value

    static let encodedRowBytes (name: string) (encoding: TensorEncoding) (columns: int) =
        if columns % encoding.BlockElements <> 0 then
            raise (invalid name $"has rows of {columns} values, not whole {encoding} blocks of {encoding.BlockElements}.")

        let bytes = int64 (columns / encoding.BlockElements) * int64 encoding.BlockBytes
        if bytes > int64 Array.MaxLength then
            raise (invalid name "has rows larger than a managed array.")
        int bytes

    member _.Name = name
    member _.Encoding = encoding
    member _.Rows = rows
    member _.Columns = columns

    /// Encoded bytes per row.
    member _.RowBytes = rowBytes

    /// All rows, stored one after another.
    member _.Payload = payload

    /// Encoded bytes of one row. Raises ArgumentOutOfRangeException outside [0, Rows).
    member _.Row(index: int) : ReadOnlySpan<byte> =
        if index < 0 || index >= rows then
            raise (ArgumentOutOfRangeException(nameof index, $"Row index must be in [0, {rows})."))
        payload.Span.Slice(index * rowBytes, rowBytes)

    /// Decodes one row into exactly Columns F32 values; see RowDecoder.decode. Raises
    /// ArgumentOutOfRangeException outside [0, Rows) and ArgumentException for another length.
    member this.DecodeRow(index: int, destination: Span<single>) =
        if destination.Length <> columns then
            invalidArg (nameof destination) $"Row {index} of '{name}' decodes into {columns} values, not {destination.Length}."
        RowDecoder.decode (encoding, this.Row(index), destination)

    /// Reads a rank-2 tensor from a source, checking its shape before reading the payload.
    /// Raises KeyNotFoundException for an unknown name. Raises InvalidDataException for a tensor
    /// without a runtime encoding, another rank, an empty dimension, rows that do not hold whole
    /// quantization blocks, a size beyond a managed array, or a payload that does not match the shape.
    static member Read(source: ITensorSource, name: string) : WeightMatrix =
        ArgumentNullException.ThrowIfNull(source)

        let descriptor =
            match source.TryFindTensor(name) with
            | Some descriptor -> descriptor
            | None -> raise (KeyNotFoundException($"Tensor '{name}' does not exist."))

        let encoding =
            match descriptor.Encoding with
            | Some encoding -> encoding
            | None -> raise (invalid name $"uses {descriptor.SourceEncoding}, which Marmotte does not represent at runtime.")

        if descriptor.Shape.Length <> 2 then
            raise (invalid name $"has rank {descriptor.Shape.Length}; a weight matrix needs rank 2.")

        let rows = dimension name "rows" descriptor.Shape[0]
        let columns = dimension name "columns" descriptor.Shape[1]
        let rowBytes = encodedRowBytes name encoding columns
        if rows > Array.MaxLength / rowBytes then
            raise (invalid name "has a payload larger than a managed array.")

        let payload = source.ReadPayload(name)
        let expectedBytes = rows * rowBytes
        if payload.Length <> expectedBytes then
            raise (invalid name $"has a {payload.Length}-byte payload; its shape needs {expectedBytes} bytes.")

        WeightMatrix(name, encoding, rows, columns, rowBytes, payload)
