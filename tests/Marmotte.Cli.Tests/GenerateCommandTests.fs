module Marmotte.Cli.GenerateCommandTests

open System.IO
open System.Text
open Marmotte.Cli.TestSupport
open Xunit

// Independently packed GGUF: identity embeddings, zero transformer matrices and
// unit norms make every continuation repeat its last input token.
let private modelBytes eos =
    use stream = new MemoryStream()
    use writer = new BinaryWriter(stream, Encoding.UTF8, true)
    let text (value: string) =
        let bytes = Encoding.UTF8.GetBytes value
        writer.Write(uint64 bytes.Length)
        writer.Write(bytes)
    let metadata = [
        "general.architecture", 8u, (fun () -> text "mistral3")
        "tokenizer.ggml.model", 8u, (fun () -> text "gpt2")
        "tokenizer.ggml.pre", 8u, (fun () -> text "tekken")
        "tokenizer.ggml.add_bos_token", 7u, (fun () -> writer.Write(false))
        "tokenizer.ggml.eos_token_id", 4u, (fun () -> writer.Write(eos: uint32))
        "tokenizer.ggml.tokens", 9u, (fun () ->
            writer.Write(8u)
            writer.Write(4UL)
            for token in [ "a"; "b"; "["; "]" ] do text token)
        for suffix, value in [ "context_length", 8u; "block_count", 1u; "embedding_length", 4u;
                              "feed_forward_length", 4u; "attention.head_count", 2u;
                              "attention.head_count_kv", 2u; "attention.key_length", 2u;
                              "attention.value_length", 2u ] do
            "mistral3." + suffix, 4u, (fun () -> writer.Write(value))
        "mistral3.attention.layer_norm_rms_epsilon", 6u, (fun () -> writer.Write(1e-5f))
        "mistral3.rope.freq_base", 6u, (fun () -> writer.Write(10000.0f))
    ]
    let tensors = [
        "token_embd.weight", [| 4UL; 4UL |], Array.init 16 (fun i -> if i / 4 = i % 4 then 1.0f else 0.0f)
        "output_norm.weight", [| 4UL |], Array.create 4 1.0f
        for name in [ "attn_norm"; "ffn_norm" ] do
            "blk.0." + name + ".weight", [| 4UL |], Array.create 4 1.0f
        for name in [ "attn_q"; "attn_k"; "attn_v"; "attn_output"; "ffn_gate"; "ffn_up"; "ffn_down" ] do
            "blk.0." + name + ".weight", [| 4UL; 4UL |], Array.zeroCreate<single> 16
    ]
    writer.Write(Encoding.ASCII.GetBytes "GGUF")
    writer.Write(3u)
    writer.Write(uint64 tensors.Length)
    writer.Write(uint64 metadata.Length)
    for name, tag, write in metadata do
        text name
        writer.Write(tag)
        write ()
    let mutable offset = 0UL
    for name, shape, values in tensors do
        text name
        writer.Write(uint32 shape.Length)
        for dimension in shape do writer.Write(dimension)
        writer.Write(0u)
        writer.Write(offset)
        offset <- (offset + uint64 values.Length * 4UL + 31UL) / 32UL * 32UL
    while stream.Position % 32L <> 0L do writer.Write(0uy)
    for _, _, values in tensors do
        for value in values do writer.Write(value)
        while stream.Position % 32L <> 0L do writer.Write(0uy)
    stream.ToArray()

[<Theory>]
[<InlineData("a", 3, "aaa", "TokenLimit")>]
[<InlineData("[", 2, "[[", "TokenLimit")>]
[<InlineData("a", 0, "", "TokenLimit")>]
[<InlineData("]", 3, "", "EndToken")>]
let ``streams literal continuation and stops on budget or EOS`` prompt count expected reason =
    use model = new ModelFile(modelBytes 3u)
    let result = run [ "generate"; model.Path; prompt; "--max-tokens"; string count ]
    Assert.Equal(0, result.ExitCode)
    Assert.Equal(expected, result.Output)
    Assert.Contains(reason, result.Error)

[<Fact>]
let ``context exhaustion stops generation`` () =
    use model = new ModelFile(modelBytes 3u)
    let result = run [ "generate"; model.Path; "a"; "--context-size"; "1" ]
    Assert.Equal(0, result.ExitCode)
    Assert.Equal("a", result.Output)
    Assert.Contains("ContextLimit", result.Error)

[<Theory>]
[<InlineData("--max-tokens", "-1")>]
[<InlineData("--context-size", "0")>]
[<InlineData("--batch-size", "0")>]
let ``invalid limits fail before opening a model`` option value =
    let result = run [ "generate"; "missing.gguf"; "a"; option; value ]
    Assert.NotEqual(0, result.ExitCode)
    Assert.Contains(option, result.Output + result.Error)

[<Fact>]
let ``oversized prompt fails without emitting text`` () =
    use model = new ModelFile(modelBytes 3u)
    let result = run [ "generate"; model.Path; "ab"; "--context-size"; "1" ]
    Assert.NotEqual(0, result.ExitCode)
    Assert.Empty(result.Output)
    Assert.Contains("encoded prompt", result.Error)

[<Fact>]
let ``reports load time before the stop summary and generation metrics after`` () =
    use model = new ModelFile(modelBytes 3u)
    let result = run [ "generate"; model.Path; "a"; "--max-tokens"; "3" ]
    Assert.Equal(0, result.ExitCode)
    Assert.Equal("aaa", result.Output)
    Assert.Matches(@"Load: \d+\.\d{3} s", result.Error)
    Assert.Matches(@"Timing: prompt processing \d+\.\d{3} s \(1 prompt tokens\); generation \d+\.\d{3} s for 3 tokens \(\d+\.\d tokens/s\)\.",
                   result.Error)
    Assert.True(result.Error.IndexOf("Load:") < result.Error.IndexOf("Stopped:"),
                "The load time must be reported before prompt processing starts.")

[<Fact>]
let ``reports zero generated tokens without a rate when the budget is zero`` () =
    use model = new ModelFile(modelBytes 3u)
    let result = run [ "generate"; model.Path; "a"; "--max-tokens"; "0" ]
    Assert.Equal(0, result.ExitCode)
    Assert.Empty(result.Output)
    Assert.Matches(@"Load: \d+\.\d{3} s", result.Error)
    Assert.Matches(@"Timing: prompt processing \d+\.\d{3} s \(1 prompt tokens\); generation 0 tokens\.",
                   result.Error)
