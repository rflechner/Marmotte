namespace Marmotte.Gguf

open System
open System.Collections.Generic
open System.Globalization
open System.IO
open Marmotte.Core

/// Associates GGUF tensor names with the dense model roles described in docs/model-configuration.md.
[<RequireQualifiedAccess>]
module GgufModelTensorLayout =
    module private Names =
        [<Literal>]
        let TokenEmbedding = "token_embd.weight"

        [<Literal>]
        let OutputNormalization = "output_norm.weight"

        [<Literal>]
        let Output = "output.weight"

        [<Literal>]
        let AttentionNormalization = "attn_norm.weight"

        [<Literal>]
        let Query = "attn_q.weight"

        [<Literal>]
        let Key = "attn_k.weight"

        [<Literal>]
        let Value = "attn_v.weight"

        [<Literal>]
        let AttentionOutput = "attn_output.weight"

        [<Literal>]
        let FeedForwardNormalization = "ffn_norm.weight"

        [<Literal>]
        let FeedForwardGate = "ffn_gate.weight"

        [<Literal>]
        let FeedForwardUp = "ffn_up.weight"

        [<Literal>]
        let FeedForwardDown = "ffn_down.weight"

    let private fail name message =
        raise (InvalidDataException($"Model tensor '{name}': {message}"))

    let private shapeText (shape: uint64 array) =
        shape |> Array.map string |> String.concat "; " |> sprintf "[%s]"

    /// Read configuration and map a parsed GGUF index, without accessing tensor payloads.
    /// Supports dense mistral3 with separate Q/K/V matrices, equal key/value head lengths,
    /// no biases and no additional tensors. Missing output.weight ties output to embeddings.
    /// Shapes are copied and reversed into Core's row-major order; unknown encodings remain
    /// unknown. This validates roles and shapes, not runtime encoding support or a tokenizer.
    /// Raises InvalidDataException for invalid configuration, missing/duplicate/unexpected
    /// tensors or incompatible shapes, and ArgumentNullException for a null index.
    /// Input descriptors must come from the GGUF reader; payload bounds remain its responsibility.
    let read (model: GgufFile) : ModelTensorLayout =
        ArgumentNullException.ThrowIfNull(model)
        let configuration = GgufModelConfiguration.read model.Metadata
        if configuration.KeyLength <> configuration.ValueLength then
            fail Names.Value "mistral3 requires equal key and value head lengths."

        // Bound allocation by the index already read, even if block_count is maliciously large.
        let minimumTensorCount = 2L + 9L * int64 configuration.BlockCount
        if int64 model.Tensors.Length < minimumTensorCount then
            raise (InvalidDataException($"Model needs at least {minimumTensorCount} tensors for {configuration.BlockCount} blocks; index contains {model.Tensors.Length}."))

        let tensors = Dictionary<string, GgufTensorInfo>(StringComparer.Ordinal)
        for tensor in model.Tensors do
            if not (tensors.TryAdd(tensor.Name, tensor)) then
                fail tensor.Name "duplicate name."
        let used = HashSet<string>(StringComparer.Ordinal)
        let find name =
            match tensors.TryGetValue name with
            | true, tensor ->
                used.Add name |> ignore
                GgufTensorSource.descriptor tensor
            | _ -> fail name "required weight is missing."
        let requireShape name expected =
            let descriptor = find name
            if descriptor.Shape <> expected then
                fail name $"expected row-major shape {shapeText expected}, found {shapeText descriptor.Shape}."
            descriptor

        let embedding = find Names.TokenEmbedding
        let embeddingWidth = uint64 configuration.EmbeddingLength
        if embedding.Shape.Length <> 2 || embedding.Shape[1] <> embeddingWidth then
            fail Names.TokenEmbedding $"expected row-major shape [vocabulary; {embeddingWidth}]."
        let vocabularySize = embedding.Shape[0]
        if vocabularySize = 0UL || vocabularySize > uint64 Int32.MaxValue then
            fail Names.TokenEmbedding "vocabulary size must be positive and fit Int32."
        let outputNormalization = requireShape Names.OutputNormalization [| embeddingWidth |]
        let hasTiedOutput = not (tensors.ContainsKey Names.Output)
        let output =
            if hasTiedOutput then embedding
            else requireShape Names.Output [| vocabularySize; embeddingWidth |]

        let queryWidth = uint64 configuration.AttentionHeadCount * uint64 configuration.KeyLength
        let keyValueWidth = uint64 configuration.KeyValueHeadCount * uint64 configuration.KeyLength
        let feedForwardWidth = uint64 configuration.FeedForwardLength
        let layer (index: int) =
            let prefix = "blk." + index.ToString(CultureInfo.InvariantCulture) + "."
            let weight suffix shape = requireShape (prefix + suffix) shape
            { AttentionNormalization = weight Names.AttentionNormalization [| embeddingWidth |]
              Query = weight Names.Query [| queryWidth; embeddingWidth |]
              Key = weight Names.Key [| keyValueWidth; embeddingWidth |]
              Value = weight Names.Value [| keyValueWidth; embeddingWidth |]
              AttentionOutput = weight Names.AttentionOutput [| embeddingWidth; queryWidth |]
              FeedForwardNormalization = weight Names.FeedForwardNormalization [| embeddingWidth |]
              FeedForwardGate = weight Names.FeedForwardGate [| feedForwardWidth; embeddingWidth |]
              FeedForwardUp = weight Names.FeedForwardUp [| feedForwardWidth; embeddingWidth |]
              FeedForwardDown = weight Names.FeedForwardDown [| embeddingWidth; feedForwardWidth |] }
        let layers = Array.init configuration.BlockCount layer
        for tensor in model.Tensors do
            if not (used.Contains tensor.Name) then
                fail tensor.Name "unexpected weight; only the documented dense, bias-free tensor layout is supported."

        { Configuration = configuration
          VocabularySize = int vocabularySize
          TokenEmbedding = embedding
          OutputNormalization = outputNormalization
          Output = output
          HasTiedOutput = hasTiedOutput
          Layers = Array.AsReadOnly layers }
