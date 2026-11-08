namespace Marmotte.Core

open System
open System.Collections.Generic
open System.IO
open System.Text

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
module private BpeByteAlphabet =
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

/// Safe, synchronous BPE over pre-tokenized UTF-8 pieces. Construction copies caller
/// arrays. Encoding uses local state and may run concurrently. No normalization,
/// pre-tokenization, text decoding or implicit recognition of special spellings occurs.
/// The reference implementation rescans adjacent pairs after every merge; it is
/// intended for correctness before performance tuning, not unbounded input pieces.
type BpeTokenizer(tokens: BpeToken array, merges: BpeMerge array,
                  boundaries: TokenBoundaryPolicy, preferWholePiece: bool) =
    let utf8 = UTF8Encoding(false, true)
    let vocabulary =
        ArgumentNullException.ThrowIfNull(tokens)
        Array.copy tokens
    let ranks = Dictionary<struct (string * string), int>()
    let ordinaryIds = Dictionary<string, int>(StringComparer.Ordinal)
    let isOrdinary category =
        category = TokenCategory.Normal || category = TokenCategory.Byte
    let validateId name id =
        if id < 0 || id >= vocabulary.Length then
            invalidArg name "Token ID is outside the vocabulary."

    do
        ArgumentNullException.ThrowIfNull(merges)
        ArgumentNullException.ThrowIfNull(boundaries)
        if vocabulary.Length = 0 then invalidArg "tokens" "Vocabulary must not be empty."
        let spellings = HashSet<string>(StringComparer.Ordinal)
        for id in 0 .. vocabulary.Length - 1 do
            let token = vocabulary[id]
            if isNull (box token) || String.IsNullOrEmpty token.Spelling then
                invalidArg "tokens" "Token entries must have non-empty spellings."
            if not (spellings.Add token.Spelling) then
                invalidArg "tokens" "Duplicate token spelling."
            if isOrdinary token.Category then ordinaryIds.Add(token.Spelling, id)

        for rank in 0 .. merges.Length - 1 do
            let merge = merges[rank]
            if isNull (box merge) || String.IsNullOrEmpty merge.Left || String.IsNullOrEmpty merge.Right then
                invalidArg "merges" "Merge operands must not be empty."
            for spelling in [ merge.Left; merge.Right; merge.Left + merge.Right ] do
                if not (ordinaryIds.ContainsKey spelling) then
                    invalidArg "merges" "Merge operands and result must be ordinary vocabulary entries."
            if not (ranks.TryAdd(struct (merge.Left, merge.Right), rank)) then
                invalidArg "merges" "Duplicate merge pair."

        for name, id, enabled in
            [ "BosTokenId", boundaries.BosTokenId, boundaries.AddBos
              "EosTokenId", boundaries.EosTokenId, boundaries.AddEos ] do
            match id with
            | Some value -> validateId name value
            | None when enabled -> invalidArg "boundaries" $"{name} is required when insertion is enabled."
            | None -> ()

    let encodePiece (piece: string) =
        ArgumentNullException.ThrowIfNull(piece)
        let symbols = ResizeArray<string>(utf8.GetBytes(piece) |> Array.map (fun value -> BpeByteAlphabet.symbols[int value]))
        let whole = String.Concat symbols
        match ordinaryIds.TryGetValue whole with
        | true, id when preferWholePiece -> [| id |]
        | _ ->
            let mutable merging = true
            while merging do
                let mutable bestIndex = -1
                let mutable bestRank = Int32.MaxValue
                for index in 0 .. symbols.Count - 2 do
                    match ranks.TryGetValue(struct (symbols[index], symbols[index + 1])) with
                    | true, rank when rank < bestRank ->
                        bestIndex <- index
                        bestRank <- rank
                    | _ -> ()
                if bestIndex < 0 then merging <- false
                else
                    symbols[bestIndex] <- symbols[bestIndex] + symbols[bestIndex + 1]
                    symbols.RemoveAt(bestIndex + 1)
            symbols
            |> Seq.map (fun symbol ->
                match ordinaryIds.TryGetValue symbol with
                | true, id -> id
                | _ -> raise (InvalidDataException("Vocabulary cannot encode an input byte symbol.")))
            |> Seq.toArray

    /// Number of entries, including special and unused entries.
    member _.VocabularySize = vocabulary.Length

    /// Immutable boundary policy captured during construction.
    member _.Boundaries = boundaries

    /// Return an immutable entry; raises ArgumentException for an out-of-range ID.
    member _.GetToken(id: int) =
        validateId "id" id
        vocabulary[id]

    /// Encode segmented input and optionally apply configured BOS/EOS insertion once.
    /// Explicit special IDs must denote Control, UserDefined or Unknown entries.
    /// Empty input still receives configured boundaries. No duplicate-marker suppression
    /// occurs: the caller owns segmentation and must not supply a second boundary.
    /// Raises ArgumentException for invalid special IDs, EncoderFallbackException for
    /// malformed UTF-16, and InvalidDataException for missing byte coverage. Returns
    /// a fresh array; failures publish no partial result. Null inputs are rejected.
    member _.Encode(segments: TokenizerSegment array, addBoundaryTokens: bool) =
        ArgumentNullException.ThrowIfNull(segments)
        let output = ResizeArray<int>()
        if addBoundaryTokens && boundaries.AddBos then output.Add boundaries.BosTokenId.Value
        for segment in segments do
            match segment with
            | TokenizerSegment.TextPiece piece -> output.AddRange(encodePiece piece)
            | TokenizerSegment.SpecialToken id ->
                validateId "segments" id
                match vocabulary[id].Category with
                | TokenCategory.Control | TokenCategory.UserDefined | TokenCategory.Unknown -> output.Add id
                | _ -> invalidArg "segments" "Explicit special ID must identify a special token."
        if addBoundaryTokens && boundaries.AddEos then output.Add boundaries.EosTokenId.Value
        output.ToArray()
