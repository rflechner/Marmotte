namespace Marmotte.Gguf

open System
open System.IO
open System.Threading

/// Native, streaming conversion from floating-point GGUF weights to Q4_0 or Q8_0.
module GgufQuantizer =
    let private invalid message = raise (InvalidDataException message)

    let private canQuantize (tensor: GgufTensorInfo) =
        tensor.Dimensions.Length >= 2
        && tensor.ElementCount > 0UL
        && tensor.Dimensions[0] % 32UL = 0UL

    let private plan format (model: GgufFile) =
        if model.Endianness <> GgufEndianness.LittleEndian then
            invalid "Native quantization currently supports only little-endian GGUF input."
        if model.Metadata |> Map.exists (fun key _ -> key.StartsWith("split.", StringComparison.Ordinal)) then
            invalid "Split GGUF files are not supported; use a complete, single-file model."

        let targetType, _, blockBytes = QuantizationCodec.layout format
        let mutable nextOffset = 0L
        let tensors = model.Tensors |> Array.map (fun tensor ->
            if QuantizationCodec.inputElementSize tensor.TensorType |> Option.isNone then
                invalid $"Tensor '{tensor.Name}' uses {tensor.TensorType}. Only F32, F16 and BF16 input is supported; requantization is not implemented."
            let tensorType, byteSize =
                if canQuantize tensor then
                    targetType, tensor.ElementCount / 32UL * uint64 blockBytes
                else
                    tensor.TensorType, tensor.ByteSize.Value
            let offset = GgufWriter.align nextOffset model.Alignment
            if byteSize > uint64 (Int64.MaxValue - offset) then
                invalid "The output tensor data exceeds the file size limit."
            nextOffset <- offset + int64 byteSize
            { tensor with TensorType = tensorType; ByteSize = Some byteSize
                          Offset = uint64 offset; AbsoluteOffset = 0L })

        let convertedCount = model.Tensors |> Array.filter canQuantize |> Array.length
        if convertedCount = 0 then
            invalid "No eligible tensors: quantization requires a non-empty tensor of rank 2 or greater with a contiguous dimension divisible by 32."
        tensors, convertedCount, nextOffset

    let private copyTensor (input: Stream) (output: Stream) (buffer: byte array) size (cancellationToken: CancellationToken) =
        let mutable remaining = size
        while remaining > 0UL do
            cancellationToken.ThrowIfCancellationRequested()
            let count = int (min remaining (uint64 buffer.Length))
            input.ReadExactly(buffer.AsSpan(0, count))
            output.Write(buffer, 0, count)
            remaining <- remaining - uint64 count

    let private convertTensor format (input: Stream) (output: Stream) (source: byte array)
                              (encoded: byte array) (values: single array) (tensor: GgufTensorInfo)
                              (cancellationToken: CancellationToken) =
        let elementBytes = (QuantizationCodec.inputElementSize tensor.TensorType).Value
        let _, _, blockBytes = QuantizationCodec.layout format
        let mutable remaining = tensor.ElementCount / 32UL
        while remaining > 0UL do
            cancellationToken.ThrowIfCancellationRequested()
            let blockCount = int (min remaining 512UL)
            input.ReadExactly(source.AsSpan(0, blockCount * 32 * elementBytes))
            for block in 0 .. blockCount - 1 do
                for index in 0 .. 31 do
                    values[index] <- QuantizationCodec.readValue tensor.TensorType source ((block * 32 + index) * elementBytes)
                try
                    QuantizationCodec.encodeBlock format values encoded (block * blockBytes)
                with :? InvalidDataException as error ->
                    invalid $"Tensor '{tensor.Name}': {error.Message}"
            output.Write(encoded, 0, blockCount * blockBytes)
            remaining <- remaining - uint64 blockCount

    let private writeOutput format (input: Stream) (output: Stream) (model: GgufFile)
                            tensors dataBytes (cancellationToken: CancellationToken) =
        let _, profile, _ = QuantizationCodec.layout format
        // The current block layout version; docs/gguf.md cites its source.
        let metadata =
            model.Metadata
            |> Map.add "general.file_type" (GgufValue.UInt32(uint32 profile))
            |> Map.add "general.quantization_version" (GgufValue.UInt32 2u)
        let dataOffset = GgufWriter.writeIndex output model.Alignment metadata tensors
        if dataBytes > Int64.MaxValue - dataOffset then invalid "The output exceeds the file size limit."

        let source = Array.zeroCreate<byte> (512 * 32 * 4)
        let encoded = Array.zeroCreate<byte> (512 * 34)
        let values = Array.zeroCreate<single> 32
        for index in 0 .. model.Tensors.Length - 1 do
            cancellationToken.ThrowIfCancellationRequested()
            let tensor = model.Tensors[index]
            input.Position <- tensor.AbsoluteOffset
            output.Position <- dataOffset + int64 tensors[index].Offset
            if canQuantize tensor then
                convertTensor format input output source encoded values tensor cancellationToken
            else
                copyTensor input output source tensor.ByteSize.Value cancellationToken
        output.SetLength(dataOffset + dataBytes)
        output.Flush()

        // Validate the completed index and payload ranges before publishing the file.
        output.Position <- 0L
        Gguf.read output |> ignore
        output.Length

    /// Quantize a complete little-endian v2/v3 GGUF containing only F32, F16 and BF16 tensors.
    /// Non-empty tensors of rank >= 2 with dimensions[0] divisible by 32 are converted;
    /// scalars, vectors, empty tensors and incompatible row widths are copied byte for byte.
    /// Output is v3 little-endian, preserving metadata except file_type and quantization_version.
    /// Uses the reader's default index limits and less than 128 KiB of reusable payload buffers.
    /// Opens and closes both files. The output path must not exist, including the input itself.
    /// A temporary sibling is moved into place only on success; failures/handled cancellation remove it.
    /// Invalid/unsupported data or non-finite converted weights raise InvalidDataException;
    /// identical paths raise ArgumentException, cancellation raises OperationCanceledException,
    /// and ordinary I/O errors propagate. The output directory must already exist.
    let quantizeFileWithCancellation format inputPath outputPath (cancellationToken: CancellationToken) =
        cancellationToken.ThrowIfCancellationRequested()
        let inputPath = Path.GetFullPath inputPath
        let outputPath = Path.GetFullPath outputPath
        if inputPath = outputPath then invalidArg (nameof outputPath) "Input and output must be different files."
        if File.Exists outputPath || Directory.Exists outputPath then
            raise (IOException("The output path already exists; choose a new file."))

        use input = File.OpenRead inputPath
        let model = Gguf.read input
        let tensors, convertedCount, dataBytes = plan format model
        let temporaryPath = Path.Combine(Path.GetDirectoryName outputPath, $".marmotte-{Guid.NewGuid():N}.tmp")
        let mutable temporaryCreated = false
        try
            let outputBytes =
                use output = new FileStream(temporaryPath, FileMode.CreateNew, FileAccess.ReadWrite, FileShare.None)
                temporaryCreated <- true
                writeOutput format input output model tensors dataBytes cancellationToken
            cancellationToken.ThrowIfCancellationRequested()
            File.Move(temporaryPath, outputPath, false)
            { Format = format
              QuantizedTensors = convertedCount
              PreservedTensors = tensors.Length - convertedCount
              InputBytes = input.Length
              OutputBytes = outputBytes }
        finally
            if temporaryCreated && File.Exists temporaryPath then File.Delete temporaryPath

    /// Quantize a file without cancellation. See quantizeFileWithCancellation for supported
    /// encodings, memory use, output publication, and exception contracts.
    let quantizeFile format inputPath outputPath =
        quantizeFileWithCancellation format inputPath outputPath CancellationToken.None
