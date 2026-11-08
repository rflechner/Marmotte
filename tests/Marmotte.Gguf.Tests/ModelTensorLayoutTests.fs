module Marmotte.Gguf.ModelTensorLayoutTests

open System
open System.IO
open Marmotte.Core
open Xunit

let private tensor name dimensions =
    let elements = Array.fold (*) 1UL dimensions
    { Name = name
      Dimensions = dimensions
      TensorType = GgmlType.F32
      ElementCount = elements
      Offset = 0UL
      AbsoluteOffset = 0L
      ByteSize = Some (elements * 4UL) }

// Independent GGML-order shapes: embedding width 6, query width 8, KV width 4.
let private fixture () =
    let metadata =
        [ "general.architecture", GgufValue.String "mistral3"
          "mistral3.context_length", GgufValue.UInt32 32u
          "mistral3.block_count", GgufValue.UInt32 2u
          "mistral3.embedding_length", GgufValue.UInt32 6u
          "mistral3.feed_forward_length", GgufValue.UInt32 10u
          "mistral3.attention.head_count", GgufValue.UInt32 4u
          "mistral3.attention.head_count_kv", GgufValue.UInt32 2u
          "mistral3.attention.key_length", GgufValue.UInt32 2u
          "mistral3.attention.value_length", GgufValue.UInt32 2u
          "mistral3.attention.layer_norm_rms_epsilon", GgufValue.Float32 0.00001f
          "mistral3.rope.freq_base", GgufValue.Float32 10000.0f ] |> Map.ofList
    let tensors =
        [| tensor "token_embd.weight" [| 6UL; 13UL |]
           tensor "output_norm.weight" [| 6UL |]
           for block in [ 1; 0 ] do
               for suffix, shape in
                   [ "attn_norm.weight", [| 6UL |]
                     "attn_q.weight", [| 6UL; 8UL |]
                     "attn_k.weight", [| 6UL; 4UL |]
                     "attn_v.weight", [| 6UL; 4UL |]
                     "attn_output.weight", [| 8UL; 6UL |]
                     "ffn_norm.weight", [| 6UL |]
                     "ffn_gate.weight", [| 6UL; 10UL |]
                     "ffn_up.weight", [| 6UL; 10UL |]
                     "ffn_down.weight", [| 10UL; 6UL |] ] do
                   tensor $"blk.{block}.{suffix}" shape |]
    { Version = 3u
      Endianness = GgufEndianness.LittleEndian
      Metadata = metadata
      Tensors = tensors
      Alignment = 32u
      DataOffset = 0L }

[<Fact>]
let ``Maps every layer role in block order and ties missing output`` () =
    let model = fixture ()
    let layout = GgufModelTensorLayout.read model
    Assert.Equal(13, layout.VocabularySize)
    Assert.Equal(2, layout.Layers.Count)
    Assert.True(layout.HasTiedOutput)
    Assert.Same(layout.TokenEmbedding, layout.Output)
    Assert.Equal<uint64>([| 13UL; 6UL |], layout.TokenEmbedding.Shape)
    for index in 0 .. 1 do
        let layer = layout.Layers[index]
        let expected =
            [ layer.AttentionNormalization, "attn_norm.weight", [| 6UL |]
              layer.Query, "attn_q.weight", [| 8UL; 6UL |]
              layer.Key, "attn_k.weight", [| 4UL; 6UL |]
              layer.Value, "attn_v.weight", [| 4UL; 6UL |]
              layer.AttentionOutput, "attn_output.weight", [| 6UL; 8UL |]
              layer.FeedForwardNormalization, "ffn_norm.weight", [| 6UL |]
              layer.FeedForwardGate, "ffn_gate.weight", [| 10UL; 6UL |]
              layer.FeedForwardUp, "ffn_up.weight", [| 10UL; 6UL |]
              layer.FeedForwardDown, "ffn_down.weight", [| 6UL; 10UL |] ]
        for descriptor, suffix, shape in expected do
            Assert.Equal($"blk.{index}.{suffix}", descriptor.Name)
            Assert.Equal<uint64>(shape, descriptor.Shape)
    model.Tensors[0].Dimensions[0] <- 99UL
    Assert.Equal(6UL, layout.TokenEmbedding.Shape[1])

[<Fact>]
let ``Explicit output is validated and retained`` () =
    let model = fixture ()
    let output = tensor "output.weight" [| 6UL; 13UL |]
    let layout = GgufModelTensorLayout.read { model with Tensors = Array.append model.Tensors [| output |] }
    Assert.False(layout.HasTiedOutput)
    Assert.Equal("output.weight", layout.Output.Name)
    let wrong = { output with Dimensions = [| 6UL; 12UL |] }
    Assert.Throws<InvalidDataException>(fun () ->
        GgufModelTensorLayout.read { model with Tensors = Array.append model.Tensors [| wrong |] } |> ignore) |> ignore

[<Fact>]
let ``Every required role rejects missing names and wrong dimensions`` () =
    let model = fixture ()
    for original in model.Tensors do
        for replacement in
            [ { original with Name = original.Name + ".unexpected" }
              { original with Dimensions = [| 1UL; 1UL; 1UL |] }
              { original with Dimensions = original.Dimensions |> Array.map (fun value -> value + 1UL) } ] do
            let tensors = model.Tensors |> Array.map (fun value -> if value.Name = original.Name then replacement else value)
            let error = Assert.Throws<InvalidDataException>(fun () -> GgufModelTensorLayout.read { model with Tensors = tensors } |> ignore)
            Assert.Contains(original.Name, error.Message)

[<Fact>]
let ``Rejects duplicate unexpected and out of range tensors`` () =
    let model = fixture ()
    for extra in
        [ model.Tensors[0]
          tensor "blk.2.attn_norm.weight" [| 6UL |]
          tensor "blk.0.attn_q.bias" [| 8UL |]
          tensor "blk.0.rope_freqs.weight" [| 1UL |] ] do
        let error = Assert.Throws<InvalidDataException>(fun () ->
            GgufModelTensorLayout.read { model with Tensors = Array.append model.Tensors [| extra |] } |> ignore)
        Assert.Contains(extra.Name, error.Message)

[<Fact>]
let ``Rejects huge block counts unequal heads and invalid vocabulary sizes`` () =
    let model = fixture ()
    for key, value in [ "mistral3.block_count", 2147483647u; "mistral3.attention.value_length", 4u ] do
        Assert.Throws<InvalidDataException>(fun () ->
            GgufModelTensorLayout.read { model with Metadata = model.Metadata |> Map.add key (GgufValue.UInt32 value) } |> ignore) |> ignore
    for vocabulary in [ 0UL; uint64 Int32.MaxValue + 1UL ] do
        let tensors = Array.copy model.Tensors
        tensors[0] <- { tensors[0] with Dimensions = [| 6UL; vocabulary |] }
        Assert.Throws<InvalidDataException>(fun () -> GgufModelTensorLayout.read { model with Tensors = tensors } |> ignore) |> ignore

[<Fact>]
let ``Mapping preserves unsupported encodings without reading payloads`` () =
    let model = fixture ()
    let tensors = Array.copy model.Tensors
    tensors[0] <- { tensors[0] with TensorType = LanguagePrimitives.EnumOfValue 999u; ByteSize = None }
    let layout = GgufModelTensorLayout.read { model with Tensors = tensors }
    Assert.True(layout.TokenEmbedding.Encoding.IsNone)
    Assert.True(layout.TokenEmbedding.ByteSize.IsNone)
    Assert.Equal("999", layout.TokenEmbedding.SourceEncoding)
