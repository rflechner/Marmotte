namespace Marmotte.Core

open System
open System.Globalization
open System.Text

/// Splits ordinary text according to the tekken profile documented in docs/tekken.md.
/// This is a Unicode-category scanner, not a normalization or special-token parser.
[<RequireQualifiedAccess>]
module TekkenPreTokenizer =
    type private Character =
        { Offset: int
          Value: int
          UpperGroup: bool
          LowerGroup: bool
          Letter: bool
          Number: bool
          Whitespace: bool }

    let private readCharacters (text: string) =
        let characters = ResizeArray<Character>()
        let mutable offset = 0
        while offset < text.Length do
            let mutable rune = Unchecked.defaultof<Rune>
            if not (Rune.TryGetRuneAt(text, offset, &rune)) then
                raise (EncoderFallbackException($"Invalid UTF-16 at offset {offset}."))
            let category = Rune.GetUnicodeCategory rune
            let shared =
                match category with
                | UnicodeCategory.ModifierLetter | UnicodeCategory.OtherLetter
                | UnicodeCategory.NonSpacingMark | UnicodeCategory.SpacingCombiningMark
                | UnicodeCategory.EnclosingMark -> true
                | _ -> false
            characters.Add
                { Offset = offset
                  Value = rune.Value
                  UpperGroup = shared || category = UnicodeCategory.UppercaseLetter || category = UnicodeCategory.TitlecaseLetter
                  LowerGroup = shared || category = UnicodeCategory.LowercaseLetter
                  Letter = Rune.IsLetter rune
                  Number = Rune.IsNumber rune
                  Whitespace = Rune.IsWhiteSpace rune }
            offset <- offset + rune.Utf16SequenceLength
        characters.ToArray()

    let private isLineBreak character = character.Value = 13 || character.Value = 10

    let private canPrefixWord character =
        not (isLineBreak character || character.Letter || character.Number)

    let private isPunctuationPiece character =
        not (character.Whitespace || character.Letter || character.Number)

    let private scanWhile predicate (characters: Character array) start =
        let mutable finish = start
        while finish < characters.Length && predicate characters[finish] do
            finish <- finish + 1
        finish

    // The first word alternative requires a lower-group character. Shared categories
    // may satisfy either group, so trailing upper-only letters can force a shorter match.
    let private lowerEndingWord characters start =
        let upperEnd = scanWhile (fun character -> character.UpperGroup) characters start
        let lowerEnd = scanWhile (fun character -> character.LowerGroup) characters upperEnd
        if lowerEnd > upperEnd then lowerEnd
        else
            let mutable lastLower = upperEnd - 1
            while lastLower >= start && not characters[lastLower].LowerGroup do
                lastLower <- lastLower - 1
            if lastLower >= start then lastLower + 1 else start

    let private upperStartingWord characters start =
        let upperEnd = scanWhile (fun character -> character.UpperGroup) characters start
        if upperEnd = start then start
        else scanWhile (fun character -> character.LowerGroup) characters upperEnd

    let private wordWithPrefix matcher (characters: Character array) start =
        let prefixedEnd =
            if canPrefixWord characters[start] then matcher characters (start + 1)
            else start + 1
        if prefixedEnd > start + 1 then prefixedEnd
        else matcher characters start

    let private punctuationEnd (characters: Character array) start =
        let contentStart = if characters[start].Value = 32 then start + 1 else start
        let contentEnd = scanWhile isPunctuationPiece characters contentStart
        if contentEnd = contentStart then start
        else
            scanWhile (fun character -> isLineBreak character || character.Value = 47) characters contentEnd

    let private whitespaceEnd (characters: Character array) start =
        let finish = scanWhile (fun character -> character.Whitespace) characters start
        let mutable lastLineBreak = finish - 1
        while lastLineBreak >= start && not (isLineBreak characters[lastLineBreak]) do
            lastLineBreak <- lastLineBreak - 1
        if lastLineBreak >= start then lastLineBreak + 1
        elif finish = characters.Length then finish
        // Before non-whitespace, leave one whitespace character for a possible word prefix.
        elif finish - start > 1 then finish - 1
        else finish

    /// Return non-empty, contiguous pieces whose concatenation equals the original text.
    /// Uses .NET Rune Unicode categories, preserves UTF-16 substrings exactly and treats
    /// each Unicode number as one piece. No trimming, normalization, byte mapping or
    /// recognition of special-token spellings occurs. Empty text returns an empty array.
    /// Raises ArgumentNullException for null and EncoderFallbackException for malformed
    /// UTF-16, before publishing any result. Allocates O(input length) working storage;
    /// calls share no mutable state. Only the tekken profile, not tekken-num3, is supported.
    let split (text: string) : string array =
        ArgumentNullException.ThrowIfNull(text)
        let characters = readCharacters text
        let pieces = ResizeArray<string>()
        let mutable start = 0
        while start < characters.Length do
            let lowerEnd = wordWithPrefix lowerEndingWord characters start
            let upperEnd =
                if lowerEnd > start then lowerEnd
                else wordWithPrefix upperStartingWord characters start
            let finish =
                if upperEnd > start then upperEnd
                elif characters[start].Number then start + 1
                else
                    let punctuation = punctuationEnd characters start
                    if punctuation > start then punctuation
                    else whitespaceEnd characters start
            if finish <= start then
                invalidOp "The pre-tokenizer did not consume a Unicode character."
            let offset = characters[start].Offset
            let endOffset = if finish = characters.Length then text.Length else characters[finish].Offset
            pieces.Add(text.Substring(offset, endOffset - offset))
            start <- finish
        pieces.ToArray()
