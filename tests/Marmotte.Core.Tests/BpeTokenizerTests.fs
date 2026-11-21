module Marmotte.Core.BpeTokenizerTests

open System
open System.IO
open System.Text
open Marmotte.Core
open Xunit

let private noBoundaries =
    { BosTokenId = None; EosTokenId = None; AddBos = false; AddEos = false }

let private normal spelling = { Spelling = spelling; Category = TokenCategory.Normal }
let private pair left right = { Left = left; Right = right }
let private encode (tokenizer: BpeTokenizer) pieces =
    tokenizer.Encode(Array.map TokenizerSegment.TextPiece pieces, false)

[<Fact>]
let ``rank beats position and equal rank chooses leftmost pair`` () =
    let tokens = Array.map normal [| "a"; "b"; "c"; "ab"; "bc"; "aa" |]
    let tokenizer = BpeTokenizer(tokens, [| pair "b" "c"; pair "a" "b"; pair "a" "a" |], noBoundaries, false)
    Assert.Equal<int>([| 0; 4 |], encode tokenizer [| "abc" |])
    Assert.Equal<int>([| 5; 0 |], encode tokenizer [| "aaa" |])

[<Fact>]
let ``cat example merges repeatedly without crossing piece boundaries`` () =
    let tokens = Array.map normal [| "c"; "a"; "t"; "ca"; "cat" |]
    let tokenizer = BpeTokenizer(tokens, [| pair "c" "a"; pair "ca" "t" |], noBoundaries, false)
    Assert.Equal<int>([| 4; 4 |], encode tokenizer [| "catcat" |])
    Assert.Equal<int>([| 0; 1; 2 |], encode tokenizer [| "c"; "at" |])

[<Fact>]
let ``byte alphabet preserves spaces nul newline accents and supplementary characters`` () =
    // Independently specified byte alphabet spellings for UTF-8 bytes:
    // 20, 00, 0A, C3 A9, F0 9F 98 80.
    let tokens = Array.map normal [| "Ġ"; "Ā"; "Ċ"; "Ã"; "©"; "ð"; "Ł"; "ĺ"; "Ģ" |]
    let tokenizer = BpeTokenizer(tokens, [||], noBoundaries, false)
    Assert.Equal<int>([| 0; 1; 2; 3; 4; 5; 6; 7; 8 |], encode tokenizer [| " \u0000\né😀" |])
    Assert.Throws<EncoderFallbackException>(fun () -> encode tokenizer [| string (char 0xD800) |] |> ignore) |> ignore

[<Fact>]
let ``whole piece preference is explicit`` () =
    let tokens = Array.map normal [| "a"; "b"; "ab" |]
    Assert.Equal<int>([| 0; 1 |], encode (BpeTokenizer(tokens, [||], noBoundaries, false)) [| "ab" |])
    Assert.Equal<int>([| 2 |], encode (BpeTokenizer(tokens, [||], noBoundaries, true)) [| "ab" |])

[<Fact>]
let ``boundaries apply once including empty input and specials require explicit insertion`` () =
    let tokens =
        [| { Spelling = "<bos>"; Category = TokenCategory.Control }
           { Spelling = "<eos>"; Category = TokenCategory.Control }
           normal "a"
           { Spelling = "<custom>"; Category = TokenCategory.UserDefined } |]
    let policy = { BosTokenId = Some 0; EosTokenId = Some 1; AddBos = true; AddEos = true }
    let tokenizer = BpeTokenizer(tokens, [||], policy, true)
    Assert.Equal<int>([| 0; 1 |], tokenizer.Encode([||], true))
    Assert.Empty(encode tokenizer [| "" |])
    Assert.Equal<int>([| 0; 2; 3; 2; 1 |], tokenizer.Encode(
        [| TokenizerSegment.TextPiece "a"; TokenizerSegment.SpecialToken 3; TokenizerSegment.TextPiece "a" |], true))
    Assert.Throws<InvalidDataException>(fun () -> encode tokenizer [| "<bos>" |] |> ignore) |> ignore
    for id in [ -1; 2; 4 ] do
        Assert.Throws<ArgumentException>(fun () -> tokenizer.Encode([| TokenizerSegment.SpecialToken id |], false) |> ignore) |> ignore

[<Fact>]
let ``constructor owns arrays and rejects ambiguous vocabulary and merge rules`` () =
    let tokens = Array.map normal [| "a"; "b"; "ab" |]
    let merges = [| pair "a" "b" |]
    let tokenizer = BpeTokenizer(tokens, merges, noBoundaries, false)
    tokens[0] <- normal "z"
    merges[0] <- pair "z" "b"
    Assert.Equal<int>([| 2 |], encode tokenizer [| "ab" |])
    let original = Array.map normal [| "a"; "b"; "ab" |]
    for badMerges in [ [| pair "a" "b"; pair "a" "b" |]; [| pair "b" "a" |]; [| pair "" "a" |] ] do
        Assert.Throws<ArgumentException>(fun () -> BpeTokenizer(original, badMerges, noBoundaries, false) |> ignore) |> ignore
    Assert.Throws<ArgumentException>(fun () -> BpeTokenizer([| normal "a"; normal "a" |], [||], noBoundaries, false) |> ignore) |> ignore
    Assert.Throws<ArgumentException>(fun () -> BpeTokenizer(original, [||], { noBoundaries with AddBos = true }, false) |> ignore) |> ignore
    Assert.Throws<InvalidDataException>(fun () -> encode tokenizer [| "z" |] |> ignore) |> ignore

[<Fact>]
let ``sentence example preserves every space and punctuation mark`` () =
    let final = [| "Hello"; ","; "Ġ"; "how"; "Ġare"; "Ġyou"; "?" |]
    let initial = [| "H"; "e"; "l"; "o"; "h"; "w"; "a"; "r"; "y"; "u"; "He"; "Hel"; "Hell"; "ho"; "ar"; "are"; "yo"; "you" |]
    let tokens = Array.append final initial |> Array.map normal
    let merges =
        [| pair "H" "e"; pair "He" "l"; pair "Hel" "l"; pair "Hell" "o"
           pair "h" "o"; pair "ho" "w"; pair "a" "r"; pair "ar" "e"
           pair "Ġ" "are"; pair "y" "o"; pair "yo" "u"; pair "Ġ" "you" |]
    let tokenizer = BpeTokenizer(tokens, merges, noBoundaries, false)
    Assert.Equal<int>([| 0; 1; 2; 3; 4; 5; 2; 6 |], encode tokenizer [| "Hello"; ","; " how"; " are"; " you"; " ?" |])
