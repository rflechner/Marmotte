namespace Marmotte.Core

open System
open System.IO
open System.Text

/// Rendering policy for Control, UserDefined and Unknown vocabulary entries.
/// Ordinary and Byte entries always use the reversible byte alphabet; Unused is rejected.
[<RequireQualifiedAccess>]
type SpecialTokenDecoding =
    | Skip
    | Include

[<RequireQualifiedAccess>]
type private TokenDecoderState =
    | Active
    | Completed
    | Faulted

/// Stateful, strict UTF-8 decoding for one token stream. Create through
/// BpeTokenizer.CreateDecoder. Instances are not thread-safe; independent instances
/// share only the tokenizer's immutable vocabulary. No disposal is required.
type TokenTextDecoder internal (getToken: int -> BpeToken, specialTokens: SpecialTokenDecoding) =
    let utf8 = UTF8Encoding(false, true)
    let decoder = utf8.GetDecoder()
    let characters = Array.zeroCreate<char> 1024
    let mutable state = TokenDecoderState.Active

    let ensureActive () =
        match state with
        | TokenDecoderState.Active -> ()
        | TokenDecoderState.Completed -> invalidOp "The token stream is complete. Reset before reuse."
        | TokenDecoderState.Faulted -> invalidOp "The token stream failed UTF-8 validation. Reset before reuse."

    let tokenBytes token =
        match token.Category with
        | TokenCategory.Normal | TokenCategory.Byte -> BpeByteAlphabet.decode token.Spelling
        | TokenCategory.Unused -> raise (InvalidDataException("Unused vocabulary entries cannot be decoded."))
        | TokenCategory.Control | TokenCategory.UserDefined | TokenCategory.Unknown ->
            match specialTokens with
            | SpecialTokenDecoding.Skip -> Array.empty<byte>
            | SpecialTokenDecoding.Include -> utf8.GetBytes token.Spelling

    let convert (bytes: byte array) flush =
        let output = StringBuilder()
        let mutable offset = 0
        let mutable completed = false
        try
            while not completed do
                let mutable bytesUsed = 0
                let mutable charsUsed = 0
                decoder.Convert(bytes, offset, bytes.Length - offset, characters, 0, characters.Length,
                                flush, &bytesUsed, &charsUsed, &completed)
                output.Append(characters, 0, charsUsed) |> ignore
                offset <- offset + bytesUsed
            output.ToString()
        with :? DecoderFallbackException ->
            // A decoder can have consumed part of this token before detecting an error.
            // Do not allow subsequent calls to build on that partially consumed state.
            state <- TokenDecoderState.Faulted
            reraise ()

    /// Consume one ID and return only complete Unicode characters, possibly an empty
    /// string. A UTF-8 sequence may span any number of tokens; at most three pending
    /// bytes are retained. No normalization, trimming or automatic stopping on EOS.
    /// Special tokens are skipped or rendered as literal UTF-8 according to the policy.
    /// ArgumentException (invalid ID), InvalidDataException (unused entry or invalid
    /// alphabet), and EncoderFallbackException (malformed literal special spelling)
    /// leave the state unchanged. DecoderFallbackException faults the stream and returns
    /// no text from this call; earlier returned text cannot be retracted. Reset is then
    /// required. Calling after completion or a UTF-8 failure raises InvalidOperationException.
    member _.Append(tokenId: int) =
        ensureActive ()
        let bytes = getToken tokenId |> tokenBytes
        convert bytes false

    /// Finish the stream, validating any pending bytes. Raises DecoderFallbackException
    /// for a truncated final sequence and faults the stream. A successful call closes it;
    /// further Append/Complete calls require Reset. Completion never inserts EOS.
    member _.Complete() =
        ensureActive ()
        let text = convert Array.empty<byte> true
        state <- TokenDecoderState.Completed
        text

    /// Discard pending bytes and reopen a completed, faulted or active stream.
    /// Does not change the vocabulary or special-token policy.
    member _.Reset() =
        decoder.Reset()
        state <- TokenDecoderState.Active
