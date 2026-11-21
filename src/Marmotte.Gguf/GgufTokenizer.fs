namespace Marmotte.Gguf

open System
open System.IO
open Marmotte.Core

/// Metadata adapter for a byte-level BPE vocabulary, with raw-text encoding for tekken.
type GgufTokenizer =
    { Tokenizer: BpeTokenizer
      /// None preserves an absent profile; the merge-only policy is then used.
      PreTokenizer: string option
      /// None preserves absence, even when the effective policy has a default.
      AddBosToken: bool option
      AddEosToken: bool option
      UnknownTokenId: int option
      SeparatorTokenId: int option
      PaddingTokenId: int option
      /// Exact tokenizer.chat_template source; None preserves absence. Parsed only on rendering.
      ChatTemplate: string option }

    /// Render a raw prompt or a text-only conversation using tokenizer.chat_template.
    /// Exposes configured BOS/EOS spellings only when their IDs exist; insertion flags
    /// do not add text. The template owns formatting. Missing variables remain errors.
    /// Returns text, not token IDs: EncodeText treats special spellings as ordinary text.
    /// Use explicit TokenizerSegment.SpecialToken IDs when assembling model input.
    member this.RenderPrompt(prompt: ChatPrompt) =
        let context =
            [ "bos_token", this.Tokenizer.Boundaries.BosTokenId
              "eos_token", this.Tokenizer.Boundaries.EosTokenId ]
            |> List.choose (fun (name, id) ->
                id |> Option.map (fun value -> name, JinjaValue.String (this.Tokenizer.GetToken(value).Spelling)))
            |> Map.ofList
        ChatPrompt.render this.ChatTemplate context prompt

    /// Split ordinary text with the declared tekken profile, then encode and optionally
    /// apply the configured BOS/EOS policy once. Special spellings remain literal text;
    /// use Tokenizer.Encode with explicit segments to insert control tokens.
    /// Raises NotSupportedException for other or missing profiles. Null/malformed text
    /// and vocabulary errors follow TekkenPreTokenizer.split and BpeTokenizer.Encode.
    member this.EncodeText(text: string, addBoundaryTokens: bool) =
        ArgumentNullException.ThrowIfNull(text)
        match this.PreTokenizer with
        | Some "tekken" ->
            let segments = TekkenPreTokenizer.split text |> Array.map TokenizerSegment.TextPiece
            this.Tokenizer.Encode(segments, addBoundaryTokens)
        | _ -> raise (NotSupportedException("Raw-text encoding requires the tekken pre-tokenizer profile."))

[<RequireQualifiedAccess>]
module GgufTokenizer =
    /// Read gpt2-family metadata without reading weights. Supports absent/default/gpt-2
    /// profiles and tekken; EncodeText splits text only for the tekken profile.
    /// Tekken enables whole-piece lookup and defaults BOS on; other profiles default
    /// BOS off. EOS defaults off. Explicit flags override defaults. Token types default
    /// to Normal. Missing merges mean no merges. No scores are used.
    /// Preserves optional tokenizer.chat_template source without parsing it; a present
    /// value must be a non-null String. RenderPrompt parses it on demand.
    /// Raises InvalidDataException for invalid or unsupported metadata and
    /// ArgumentNullException for a null map. See docs/tokenizer.md for limitations.
    let read (metadata: Map<string, GgufValue>) =
        ArgumentNullException.ThrowIfNull(metadata)
        let fail suffix message =
            raise (InvalidDataException($"Tokenizer metadata 'tokenizer.ggml.{suffix}': {message}"))
        let find suffix = Map.tryFind ("tokenizer.ggml." + suffix) metadata
        let stringValue suffix = function
            | GgufValue.String value when not (isNull value) -> value
            | _ -> fail suffix "expected String."
        let arrayValue suffix expected = function
            | GgufValue.Array (kind, values) when kind = expected && not (isNull values) -> values
            | _ -> fail suffix $"expected Array of {expected}."
        let required suffix =
            find suffix |> Option.defaultWith (fun () -> fail suffix "required value is missing.")
        if stringValue "model" (required "model") <> "gpt2" then
            fail "model" "only gpt2-family BPE is supported."
        let profile = find "pre" |> Option.map (stringValue "pre")
        let wholePiece =
            match profile with
            | None | Some "default" | Some "gpt-2" -> false
            | Some "tekken" -> true
            | Some value -> fail "pre" $"unsupported profile '{value}'."
        let spellings =
            required "tokens"
            |> arrayValue "tokens" GgufValueType.String
            |> Array.map (stringValue "tokens")
        let types =
            match find "token_type" with
            | None -> Array.create spellings.Length TokenCategory.Normal
            | Some value ->
                let values = arrayValue "token_type" GgufValueType.Int32 value
                if values.Length <> spellings.Length then fail "token_type" "length must match tokens."
                values |> Array.map (function
                    | GgufValue.Int32 1 -> TokenCategory.Normal
                    | GgufValue.Int32 2 -> TokenCategory.Unknown
                    | GgufValue.Int32 3 -> TokenCategory.Control
                    | GgufValue.Int32 4 -> TokenCategory.UserDefined
                    | GgufValue.Int32 5 -> TokenCategory.Unused
                    | GgufValue.Int32 6 -> TokenCategory.Byte
                    | _ -> fail "token_type" "expected supported Int32 category (1 through 6).")
        let tokens = Array.map2 (fun spelling category -> { Spelling = spelling; Category = category }) spellings types
        let merges =
            match find "merges" with
            | None -> [||]
            | Some value ->
                arrayValue "merges" GgufValueType.String value
                |> Array.map (fun entry ->
                    let text = stringValue "merges" entry
                    let separator = text.IndexOf(' ')
                    if separator <= 0 || separator = text.Length - 1 || text.IndexOf(' ', separator + 1) >= 0 then
                        fail "merges" "expected two non-empty byte-alphabet spellings separated by one space."
                    { Left = text.Substring(0, separator)
                      Right = text.Substring(separator + 1) })
        let tokenId suffix =
            find suffix |> Option.map (function
                | GgufValue.UInt32 value when uint64 value < uint64 tokens.Length -> int value
                | _ -> fail suffix "expected UInt32 within the vocabulary.")
        let flag suffix =
            find suffix |> Option.map (function
                | GgufValue.Bool value -> value
                | _ -> fail suffix "expected Bool.")
        let addBos = flag "add_bos_token"
        let addEos = flag "add_eos_token"
        let boundaries =
            { BosTokenId = tokenId "bos_token_id"
              EosTokenId = tokenId "eos_token_id"
              AddBos = Option.defaultValue wholePiece addBos
              AddEos = Option.defaultValue false addEos }
        let chatTemplate =
            match Map.tryFind "tokenizer.chat_template" metadata with
            | None -> None
            | Some (GgufValue.String value) when not (isNull value) -> Some value
            | Some _ -> raise (InvalidDataException("Tokenizer metadata 'tokenizer.chat_template': expected String."))
        let tokenizer =
            try BpeTokenizer(tokens, merges, boundaries, wholePiece)
            with :? ArgumentException as error ->
                raise (InvalidDataException("Invalid tokenizer vocabulary, merges or boundaries: " + error.Message, error))
        { Tokenizer = tokenizer
          PreTokenizer = profile
          AddBosToken = addBos
          AddEosToken = addEos
          UnknownTokenId = tokenId "unknown_token_id"
          SeparatorTokenId = tokenId "separator_token_id"
          PaddingTokenId = tokenId "padding_token_id"
          ChatTemplate = chatTemplate }
