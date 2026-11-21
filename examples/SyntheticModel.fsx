#r "../src/Marmotte.Core/bin/Debug/net10.0/Marmotte.Core.dll"
#r "../src/Marmotte.Cpu/bin/Debug/net10.0/Marmotte.Cpu.dll"

open System
open System.Buffers.Binary
open System.Collections.Generic
open Marmotte.Core
open Marmotte.Cpu

// Build first with dotnet build Marmotte.slnx. This invented model is deliberately simple:
// identity embeddings/output, unit normalization weights and zero block matrices. The residual
// path preserves each embedding, making the final logits easy to check without a model download.
let createExample () =
    let tensors = Dictionary<string, TensorDescriptor * byte array>()
    let weight name shape values =
        let payload = Array.zeroCreate<byte> (Array.length values * 4)
        values |> Array.iteri (fun i value -> BinaryPrimitives.WriteSingleLittleEndian(payload.AsSpan(i * 4, 4), value))
        let descriptor =
            { Name = name
              Encoding = Some TensorEncoding.F32
              SourceEncoding = "F32"
              Shape = Array.map uint64 shape
              ElementCount = uint64 (Array.length values)
              ByteSize = Some (uint64 payload.Length) }
        tensors.Add(name, (descriptor, payload))
        descriptor
    let zero name rows columns = weight name [| rows; columns |] (Array.zeroCreate (rows * columns))
    let norm name = weight name [| 4 |] (Array.create 4 1.0f)
    let embedding = weight "embedding" [| 4; 4 |] (Array.init 16 (fun i -> if i / 4 = i % 4 then 1.0f else 0.0f))
    let layer =
        { AttentionNormalization = norm "attentionNorm"
          Query = zero "query" 4 4
          Key = zero "key" 2 4
          Value = zero "value" 2 4
          AttentionOutput = zero "attentionOutput" 4 4
          FeedForwardNormalization = norm "feedForwardNorm"
          FeedForwardGate = zero "gate" 4 4
          FeedForwardUp = zero "up" 4 4
          FeedForwardDown = zero "down" 4 4 }
    let configuration =
        { Architecture = ModelArchitecture.Mistral3
          ContextLength = 8
          BlockCount = 1
          EmbeddingLength = 4
          FeedForwardLength = 4
          AttentionHeadCount = 2
          KeyValueHeadCount = 1
          KeyLength = 2
          ValueLength = 2
          RmsNormalizationEpsilon = 1e-5f
          RopeFrequencyBase = 10000.0
          RopeScaling = ModelRopeScaling.None
          AttentionTemperature = None }
    let layout =
        { Configuration = configuration
          VocabularySize = 4
          TokenEmbedding = embedding
          OutputNormalization = norm "outputNorm"
          Output = embedding
          HasTiedOutput = true
          Layers = Array.AsReadOnly [| layer |] }
    let source =
        { new ITensorSource with
            member _.Tensors = tensors.Values |> Seq.map fst |> Seq.toArray :> IReadOnlyList<_>
            member _.TryFindTensor name =
                match tensors.TryGetValue name with
                | true, (descriptor, _) -> Some descriptor
                | _ -> None
            member _.ReadPayload name = ReadOnlyMemory<byte>(snd tensors[name])
            member _.Dispose() = () }
    source, layout

