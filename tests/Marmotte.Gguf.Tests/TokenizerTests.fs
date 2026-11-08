module Marmotte.Gguf.TokenizerTests

open System.IO
open Marmotte.Core
open Marmotte.Gguf
open Xunit

let private key suffix = "tokenizer.ggml." + suffix
let private strings values = GgufValue.Array(GgufValueType.String, Array.map GgufValue.String values)
let private metadata =
    [ key "model", GgufValue.String "gpt2"
      key "tokens", strings [| "<bos>"; "<eos>"; "c"; "a"; "t"; "ca"; "cat" |]
      key "token_type", GgufValue.Array(GgufValueType.Int32, Array.map GgufValue.Int32 [| 3; 3; 1; 1; 1; 1; 1 |])
      key "merges", strings [| "c a"; "ca t" |]
      key "bos_token_id", GgufValue.UInt32 0u
      key "eos_token_id", GgufValue.UInt32 1u ] |> Map.ofList

[<Fact>]
let ``metadata preserves IDs and optional flags while resolving profile defaults`` () =
    let loaded = metadata |> Map.add (key "pre") (GgufValue.String "tekken") |> GgufTokenizer.read
    Assert.Equal(None, loaded.AddBosToken)
    Assert.True(loaded.Tokenizer.Boundaries.AddBos)
    Assert.Equal<int>([| 0; 6; 6 |], loaded.Tokenizer.Encode([| TokenizerSegment.TextPiece "catcat" |], true))
    let explicit = metadata |> Map.add (key "pre") (GgufValue.String "tekken") |> Map.add (key "add_bos_token") (GgufValue.Bool false) |> GgufTokenizer.read
    Assert.Equal(Some false, explicit.AddBosToken)
    Assert.False(explicit.Tokenizer.Boundaries.AddBos)
    let defaultProfile = GgufTokenizer.read metadata
    Assert.Equal(None, defaultProfile.PreTokenizer)
    Assert.False(defaultProfile.Tokenizer.Boundaries.AddBos)

[<Fact>]
let ``absent merges and categories are supported without invented token IDs`` () =
    let loaded = metadata |> Map.remove (key "merges") |> Map.remove (key "token_type") |> GgufTokenizer.read
    Assert.Equal<int>([| 2; 3; 4 |], loaded.Tokenizer.Encode([| TokenizerSegment.TextPiece "cat" |], false))
    Assert.Equal(None, loaded.UnknownTokenId)

[<Fact>]
let ``invalid typed metadata is rejected before encoding`` () =
    let invalid =
        [ "model", GgufValue.String "llama"
          "pre", GgufValue.String "unsupported"
          "tokens", strings [| "a"; "a" |]
          "token_type", GgufValue.Array(GgufValueType.Int32, [| GgufValue.Int32 1 |])
          "merges", strings [| "ca  t" |]
          "merges", strings [| "c z" |]
          "merges", strings [| "c a"; "c a" |]
          "bos_token_id", GgufValue.UInt32 7u
          "eos_token_id", GgufValue.Int32 1
          "padding_token_id", GgufValue.UInt32 100u
          "add_bos_token", GgufValue.UInt32 1u
          "tokens", GgufValue.Array(GgufValueType.String, [| GgufValue.Int32 1 |]) ]
    for suffix, value in invalid do
        Assert.Throws<InvalidDataException>(fun () -> metadata |> Map.add (key suffix) value |> GgufTokenizer.read |> ignore) |> ignore
    for suffix in [ "model"; "tokens" ] do
        Assert.Throws<InvalidDataException>(fun () -> metadata |> Map.remove (key suffix) |> GgufTokenizer.read |> ignore) |> ignore
    Assert.Throws<InvalidDataException>(fun () ->
        metadata |> Map.remove (key "bos_token_id") |> Map.add (key "add_bos_token") (GgufValue.Bool true)
        |> GgufTokenizer.read |> ignore) |> ignore
