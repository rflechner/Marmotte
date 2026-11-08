namespace Marmotte.Core

open System.Collections.Generic

/// Weight descriptors for one dense transformer block, in execution order by role.
/// Matrix shapes are row-major [output features; input features]; norms are vectors.
/// Descriptors contain no payload and retain their stored encoding.
type TransformerLayerTensors =
    { AttentionNormalization: TensorDescriptor
      Query: TensorDescriptor
      Key: TensorDescriptor
      Value: TensorDescriptor
      AttentionOutput: TensorDescriptor
      FeedForwardNormalization: TensorDescriptor
      FeedForwardGate: TensorDescriptor
      FeedForwardUp: TensorDescriptor
      FeedForwardDown: TensorDescriptor }

/// Format-neutral mapping of a model index, independent of any open file or backend.
/// VocabularySize is derived from embedding rows, not a validated tokenizer vocabulary.
/// Layers are ordered by zero-based block index. Shape arrays belong to this result and
/// must be treated as read-only. Public record construction does not validate the layout.
type ModelTensorLayout =
    { Configuration: ModelConfiguration
      VocabularySize: int
      TokenEmbedding: TensorDescriptor
      OutputNormalization: TensorDescriptor
      Output: TensorDescriptor
      /// True when the output descriptor reuses TokenEmbedding because no output weight exists.
      HasTiedOutput: bool
      Layers: IReadOnlyList<TransformerLayerTensors> }
