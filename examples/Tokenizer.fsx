#load "../src/Marmotte.Core/BpeTokenizer.fs"
#load "../src/Marmotte.Core/TekkenPreTokenizer.fs"

open Marmotte.Core

// The tutorial's independent vocabulary: intermediate IDs 2..19 and final IDs 20..26.
let normal spelling = { Spelling = spelling; Category = TokenCategory.Normal }
let intermediate =
    [| "H"; "e"; "l"; "o"; "h"; "w"; "a"; "r"; "y"; "u"
       "He"; "Hel"; "Hell"; "ho"; "ar"; "are"; "yo"; "you" |]
let final = [| "Hello"; ","; "Ġ"; "how"; "Ġare"; "Ġyou"; "?" |]
let tokens =
    Array.concat
        [ [| { Spelling = "<bos>"; Category = TokenCategory.Control }
             { Spelling = "<eos>"; Category = TokenCategory.Control } |]
          Array.map normal intermediate
          Array.map normal final ]

let pair left right = { Left = left; Right = right }
let merges =
    [| pair "H" "e"; pair "He" "l"; pair "Hel" "l"; pair "Hell" "o"
       pair "h" "o"; pair "ho" "w"; pair "a" "r"; pair "ar" "e"
       pair "Ġ" "are"; pair "y" "o"; pair "yo" "u"; pair "Ġ" "you" |]
let boundaries =
    { BosTokenId = Some 0
      EosTokenId = Some 1
      AddBos = true
      AddEos = false }
let tokenizer = BpeTokenizer(tokens, merges, boundaries, false)

// Pre-tokenization preserves the sentence exactly; BPE still uses the invented vocabulary.
let text = "Hello, how are you ?"
let pieces = TekkenPreTokenizer.split text
printfn "Pieces: %A" pieces
let ids = tokenizer.Encode(Array.map TokenizerSegment.TextPiece pieces, true)
let expected = [| 0; 20; 21; 22; 23; 24; 25; 22; 26 |]
if ids <> expected then failwithf "Unexpected token IDs: %A" ids
printfn "Input: %s" (String.concat "" pieces)
printfn "IDs with BOS: %A" ids
printfn "Stored spellings: %A" (ids |> Array.map (fun id -> (tokenizer.GetToken id).Spelling))
