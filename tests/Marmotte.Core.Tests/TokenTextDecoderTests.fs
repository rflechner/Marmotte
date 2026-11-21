module Marmotte.Core.TokenTextDecoderTests

open System
open System.IO
open System.Text
open Marmotte.Core
open Xunit

let private boundaries =
    { BosTokenId = None; EosTokenId = None; AddBos = false; AddEos = false }
let private normal spelling = { Spelling = spelling; Category = TokenCategory.Normal }
let private tokenizer tokens = BpeTokenizer(tokens, [||], boundaries, false)
let private fromSpellings spellings = tokenizer (Array.map normal spellings)
let private decode (model: BpeTokenizer) ids = model.Decode(ids, SpecialTokenDecoding.Skip)

[<Fact>]
let ``stored byte spellings reconstruct exact whitespace nul and Unicode`` () =
    // Bytes 20 00 0A C3 A9 F0 9F 98 80, independently specified.
    let model = fromSpellings [| "Ġ"; "Ā"; "Ċ"; "Ã"; "©"; "ð"; "Ł"; "ĺ"; "Ģ" |]
    Assert.Equal(" \u0000\né😀", decode model [| 0 .. 8 |])
    Assert.Equal("", decode model [||])
    Assert.Equal(" \u0000\né😀", decode model (model.Encode([| TokenizerSegment.TextPiece " \u0000\né😀" |], false)))

[<Fact>]
let ``two three and four byte scalars wait for their final byte`` () =
    // C3 A9 = é; E2 82 AC = euro sign; F0 9F 98 80 = grinning face.
    for spellings, expected in
        [ [| "Ã"; "©" |], "é"
          [| "â"; "Ĥ"; "¬" |], "€"
          [| "ð"; "Ł"; "ĺ"; "Ģ" |], "😀" ] do
        let stream = (fromSpellings spellings).CreateDecoder SpecialTokenDecoding.Skip
        for id in 0 .. spellings.Length - 2 do
            Assert.Equal("", stream.Append id)
        Assert.Equal(expected, stream.Append(spellings.Length - 1))
        Assert.Equal("", stream.Complete())

[<Fact>]
let ``all token boundaries within a mixed byte sequence produce the same text`` () =
    let spelling = "AÃ©âĤ¬ðŁĺĢZ"
    // Enumerate every partition of this fixed byte-alphabet string, independently of BPE.
    for mask in 0 .. (1 <<< (spelling.Length - 1)) - 1 do
        let pieces = ResizeArray<string>()
        let mutable start = 0
        for index in 0 .. spelling.Length - 2 do
            if mask &&& (1 <<< index) <> 0 then
                pieces.Add(spelling.Substring(start, index + 1 - start))
                start <- index + 1
        pieces.Add(spelling.Substring start)
        let model = fromSpellings (pieces.ToArray())
        let stream = model.CreateDecoder SpecialTokenDecoding.Skip
        let output = StringBuilder()
        for id in 0 .. pieces.Count - 1 do
            let text = stream.Append id
            // Every emitted chunk is itself valid UTF-16, even at supplementary scalars.
            UTF8Encoding(false, true).GetBytes(text) |> ignore
            output.Append text |> ignore
        output.Append(stream.Complete()) |> ignore
        Assert.Equal("Aé€😀Z", output.ToString())

[<Fact>]
let ``large tokens span the fixed character buffer without losing characters`` () =
    let spelling = String.replicate 1100 "AðŁĺĢ"
    let model = fromSpellings [| spelling |]
    Assert.Equal(String.replicate 1100 "A😀", decode model [| 0 |])

[<Fact>]
let ``special policy is explicit and literal spellings bypass the byte alphabet`` () =
    let tokens =
        [| normal "Ã"; normal "©"
           { Spelling = "<bos>"; Category = TokenCategory.Control }
           { Spelling = "Ġé"; Category = TokenCategory.UserDefined }
           { Spelling = "<unknown>"; Category = TokenCategory.Unknown }
           { Spelling = "<unused>"; Category = TokenCategory.Unused }
           { Spelling = "Ġ"; Category = TokenCategory.Byte } |]
    let model = tokenizer tokens
    Assert.Equal("é ", decode model [| 2; 0; 3; 4; 1; 6 |])
    Assert.Equal("<bos>Ġé<unknown> ", model.Decode([| 2; 3; 4; 6 |], SpecialTokenDecoding.Include))
    for policy in [ SpecialTokenDecoding.Skip; SpecialTokenDecoding.Include ] do
        Assert.Throws<InvalidDataException>(fun () -> model.Decode([| 5 |], policy) |> ignore) |> ignore
    let stream = model.CreateDecoder SpecialTokenDecoding.Include
    Assert.Equal("", stream.Append 0)
    Assert.Throws<DecoderFallbackException>(fun () -> stream.Append 2 |> ignore) |> ignore

[<Fact>]
let ``invalid IDs and alphabet errors preserve pending bytes`` () =
    let model = tokenizer [| normal "Ã"; normal "©"; normal "😀"; { Spelling = "unused"; Category = TokenCategory.Unused } |]
    let stream = model.CreateDecoder SpecialTokenDecoding.Skip
    Assert.Equal("", stream.Append 0)
    for id in [ -1; 4 ] do
        Assert.Throws<ArgumentException>(fun () -> stream.Append id |> ignore) |> ignore
    for id in [ 2; 3 ] do
        Assert.Throws<InvalidDataException>(fun () -> stream.Append id |> ignore) |> ignore
    Assert.Equal("é", stream.Append 1)
    Assert.Equal("", stream.Complete())
    Assert.Throws<ArgumentNullException>(fun () -> model.Decode(null, SpecialTokenDecoding.Skip) |> ignore) |> ignore

[<Fact>]
let ``invalid UTF8 faults the stream until reset`` () =
    // Lone continuation, overlong encoding, surrogate, out-of-range scalar, invalid lead.
    for spellings in
        [ [| "Ģ" |]; [| "À¯" |]; [| "íłĢ" |]; [| "ôĲĢĢ" |]; [| "ÿ" |]; [| "ÃA" |] ] do
        let model = fromSpellings (Array.append spellings [| "ok" |])
        let stream = model.CreateDecoder SpecialTokenDecoding.Skip
        Assert.Throws<DecoderFallbackException>(fun () -> stream.Append 0 |> ignore) |> ignore
        Assert.Throws<InvalidOperationException>(fun () -> stream.Append 1 |> ignore) |> ignore
        Assert.Throws<InvalidOperationException>(fun () -> stream.Complete() |> ignore) |> ignore
        stream.Reset()
        Assert.Equal("ok", stream.Append 1)
        Assert.Equal("", stream.Complete())
        Assert.Throws<DecoderFallbackException>(fun () -> decode model [| 0 |] |> ignore) |> ignore

[<Fact>]
let ``completion detects truncated sequences and closes successful streams`` () =
    for spelling in [ "Ã"; "âĤ"; "ðŁĺ" ] do
        let model = fromSpellings [| spelling; "ok" |]
        let stream = model.CreateDecoder SpecialTokenDecoding.Skip
        Assert.Equal("", stream.Append 0)
        Assert.Throws<DecoderFallbackException>(fun () -> stream.Complete() |> ignore) |> ignore
        Assert.Throws<InvalidOperationException>(fun () -> stream.Append 1 |> ignore) |> ignore
        Assert.Throws<DecoderFallbackException>(fun () -> decode model [| 0 |] |> ignore) |> ignore
        stream.Reset()
        Assert.Equal("ok", stream.Append 1)
        Assert.Equal("", stream.Complete())
        Assert.Throws<InvalidOperationException>(fun () -> stream.Complete() |> ignore) |> ignore
        Assert.Throws<InvalidOperationException>(fun () -> stream.Append 1 |> ignore) |> ignore
        stream.Reset()
        Assert.Equal("", stream.Complete())

[<Fact>]
let ``independent streams and reset do not share pending bytes`` () =
    let model = fromSpellings [| "Ã"; "©"; "ok"; "ï»¿" |]
    let first = model.CreateDecoder SpecialTokenDecoding.Skip
    let second = model.CreateDecoder SpecialTokenDecoding.Skip
    Assert.Equal("", first.Append 0)
    Assert.Equal("ok", second.Append 2)
    Assert.Equal("", second.Complete())
    Assert.Equal("é", first.Append 1)
    Assert.Equal("", first.Complete())
    first.Reset()
    Assert.Equal("", first.Append 0)
    first.Reset()
    Assert.Equal("ok", first.Append 2)
    Assert.Equal("", first.Complete())
    Assert.Equal("\uFEFF", decode model [| 3 |])

[<Fact>]
let ``malformed literal special spelling leaves pending bytes unchanged`` () =
    let model = tokenizer
                    [| normal "Ã"; normal "©"
                       { Spelling = string (char 0xD800); Category = TokenCategory.Control } |]
    let stream = model.CreateDecoder SpecialTokenDecoding.Include
    Assert.Equal("", stream.Append 0)
    Assert.Throws<EncoderFallbackException>(fun () -> stream.Append 2 |> ignore) |> ignore
    Assert.Equal("é", stream.Append 1)
    Assert.Equal("", stream.Complete())
