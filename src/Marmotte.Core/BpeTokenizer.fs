namespace Marmotte.Core

open System
open System.Collections.Generic
open System.IO
open System.Text

/// Safe, synchronous BPE over pre-tokenized UTF-8 pieces. Construction copies caller
/// arrays. Encoding uses local state and may run concurrently. No normalization,
/// pre-tokenization or implicit recognition of special spellings occurs during encoding.
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

    /// Create an independent streaming UTF-8 decoder. The policy explicitly controls
    /// Control, UserDefined and Unknown entries; Unused entries always fail. Call
    /// Complete when input ends, including after EOS, to detect truncated UTF-8.
    member this.CreateDecoder(specialTokens: SpecialTokenDecoding) =
        TokenTextDecoder(this.GetToken, specialTokens)

    /// Decode a complete sequence, using the same strict UTF-8 rules as CreateDecoder.
    /// Null arrays raise ArgumentNullException. Invalid IDs/spellings and malformed or
    /// truncated UTF-8 propagate the streaming decoder's exceptions; no partial string
    /// is returned. Empty input returns an empty string. Input is consumed synchronously.
    member this.Decode(tokenIds: int array, specialTokens: SpecialTokenDecoding) =
        ArgumentNullException.ThrowIfNull(tokenIds)
        let decoder = this.CreateDecoder specialTokens
        let output = StringBuilder()
        for tokenId in tokenIds do
            output.Append(decoder.Append tokenId) |> ignore
        output.Append(decoder.Complete()) |> ignore
        output.ToString()
