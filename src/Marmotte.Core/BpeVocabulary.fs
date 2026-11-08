namespace Marmotte.Core

open System.IO

/// Vocabulary categories; control and user-defined entries require explicit insertion.
[<RequireQualifiedAccess>]
type TokenCategory =
    | Normal
    | Unknown
    | Control
    | UserDefined
    | Unused
    | Byte

/// One vocabulary entry. Its array index, not its spelling or category, is its ID.
type BpeToken =
    { Spelling: string
      Category: TokenCategory }

/// A pair of stored byte-alphabet spellings. Array order defines merge rank.
type BpeMerge =
    { Left: string
      Right: string }

/// Boundary IDs and defaults. None means unavailable, not token zero.
type TokenBoundaryPolicy =
    { BosTokenId: int option
      EosTokenId: int option
      AddBos: bool
      AddEos: bool }

/// Input already segmented by a profile-specific pre-tokenizer. Special IDs are
/// inserted explicitly; text pieces never recognize control spellings automatically.
[<RequireQualifiedAccess>]
type TokenizerSegment =
    | TextPiece of string
    | SpecialToken of int

/// Reversible byte alphabet used by the supported BPE profiles; see docs/tokenizer.md.
module internal BpeByteAlphabet =
    let symbols =
        let mutable next = 256
        Array.init 256 (fun value ->
            let preserved =
                (value >= 33 && value <= 126)
                || (value >= 161 && value <= 172)
                || value >= 174
            if preserved then string (char value)
            else
                let symbol = string (char next)
                next <- next + 1
                symbol)

    let private bytesBySymbol =
        symbols |> Array.mapi (fun value symbol -> symbol[0], byte value) |> dict

    let decode (spelling: string) =
        let bytes = Array.zeroCreate<byte> spelling.Length
        for index in 0 .. spelling.Length - 1 do
            match bytesBySymbol.TryGetValue spelling[index] with
            | true, value -> bytes[index] <- value
            | _ ->
                raise (InvalidDataException("Token spelling contains a character outside the byte alphabet."))
        bytes
