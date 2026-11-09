namespace Marmotte.Cli

open System
open System.ComponentModel
open System.Diagnostics
open System.IO
open Marmotte.Core
open Marmotte.Cpu
open Marmotte.Gguf
open Spectre.Console
open Spectre.Console.Cli

/// CPU generation from a literal prompt, without chat-template interpretation.
type GenerateSettings() =
    inherit CommandSettings()

    [<CommandArgument(0, "<MODEL>")>]
    [<Description("Dense mistral3 GGUF with a tekken tokenizer.")>]
    member val ModelPath = "" with get, set

    [<CommandArgument(1, "<PROMPT>")>]
    [<Description("Literal text to continue; no chat template or special-token parsing.")>]
    member val Prompt = "" with get, set

    [<CommandOption("--max-tokens <COUNT>")>]
    [<Description("Maximum new tokens; zero processes only the prompt.")>]
    [<DefaultValue(128)>]
    member val MaximumTokens = 128 with get, set

    [<CommandOption("--context-size <COUNT>")>]
    [<Description("Cache capacity in tokens, capped at the model context.")>]
    [<DefaultValue(2048)>]
    member val ContextSize = 2048 with get, set

    [<CommandOption("--batch-size <COUNT>")>]
    [<Description("Prompt tokens per batch, capped at the selected context.")>]
    [<DefaultValue(32)>]
    member val BatchSize = 32 with get, set

    override this.Validate() =
        if String.IsNullOrWhiteSpace this.ModelPath || String.IsNullOrEmpty this.Prompt then
            ValidationResult.Error("A model path and a non-empty prompt are required.")
        elif this.MaximumTokens < 0 then
            ValidationResult.Error("--max-tokens must be zero or greater.")
        elif this.ContextSize <= 0 || this.BatchSize <= 0 then
            ValidationResult.Error("--context-size and --batch-size must be positive.")
        else
            ValidationResult.Success()

/// Stream greedy continuation to stdout; diagnostics go to stderr. Resources are
/// disposed on success, failure and cancellation. Only complete UTF-8 is emitted.
type GenerateCommand() =
    inherit Command<GenerateSettings>()

    override _.Execute(_, settings, cancellationToken) =
        cancellationToken.ThrowIfCancellationRequested()
        let loadWatch = Stopwatch.StartNew()
        let index = Gguf.readFile settings.ModelPath
        let layout = GgufModelTensorLayout.read index
        let tokenizer = GgufTokenizer.read index.Metadata
        if tokenizer.Tokenizer.VocabularySize <> layout.VocabularySize then
            raise (InvalidDataException("Tokenizer and model vocabulary sizes differ."))
        let promptIds = tokenizer.EncodeText(settings.Prompt, true)
        let capacity = min settings.ContextSize layout.Configuration.ContextLength
        if promptIds.Length = 0 || promptIds.Length > capacity then
            invalidArg "PROMPT" "The encoded prompt must be non-empty and fit --context-size and the model context."

        cancellationToken.ThrowIfCancellationRequested()
        use source = GgufTensorSource.openMappedFile settings.ModelPath
        use backend = new CpuBackend() :> ITensorBackend
        use model = TransformerModel.Load(backend, source, layout)
        use session = GenerationSession.Create(model, capacity, min settings.BatchSize capacity,
                                               TensorEncoding.F32, TensorEncoding.F32)
        loadWatch.Stop()
        // Report the load time as soon as it is known, before prompt processing starts.
        Console.Error.WriteLine($"Load: {ConsoleDsl.formatSeconds loadWatch.Elapsed}")
        let endIds = tokenizer.Tokenizer.Boundaries.EosTokenId |> Option.toArray
        let chooseToken (logits: ReadOnlyMemory<single>) = TokenSampling.greedy logits.Span
        let generator = TokenGenerator(session, settings.MaximumTokens, ReadOnlyMemory<int>(endIds), chooseToken)
        let decoder = tokenizer.Tokenizer.CreateDecoder SpecialTokenDecoding.Skip
        // The first emitted token arrives right after prompt processing, so it marks
        // the prefill/generation boundary. With no emitted token the whole run is prefill.
        let runWatch = Stopwatch.StartNew()
        let mutable prefillEnd = None
        let writeToken token =
            if prefillEnd.IsNone then prefillEnd <- Some runWatch.Elapsed
            Console.Out.Write(decoder.Append token)
            Console.Out.Flush()
        let result = generator.Generate(ReadOnlyMemory<int>(promptIds), onToken = writeToken,
                                        cancellationToken = cancellationToken)
        runWatch.Stop()
        // A token budget can cut through a multibyte character. Preserve complete text
        // and report that truncation rather than inserting a replacement character.
        try Console.Out.Write(decoder.Complete())
        with :? Text.DecoderFallbackException ->
            Console.Error.WriteLine("Generation ended with an incomplete UTF-8 character; trailing bytes were omitted.")
        Console.Out.Flush()
        Console.Error.WriteLine($"Stopped: {result.StopReason}; generated {result.Tokens.Length} tokens.")
        let prefill = match prefillEnd with Some elapsed -> elapsed | None -> runWatch.Elapsed
        let generation = runWatch.Elapsed - prefill
        let generationReport =
            if result.Tokens.Length > 0 then
                let rate =
                    if generation.TotalSeconds > 0.0 then
                        sprintf " (%s)" (ConsoleDsl.formatRate (float result.Tokens.Length / generation.TotalSeconds))
                    else ""
                $"generation {ConsoleDsl.formatSeconds generation} for {result.Tokens.Length} tokens{rate}"
            else
                "generation 0 tokens"
        Console.Error.WriteLine($"Timing: prompt processing {ConsoleDsl.formatSeconds prefill} ({promptIds.Length} prompt tokens); {generationReport}.")
        match result.StopReason with
        | GenerationStopReason.SamplingFailed _ -> 1
        | _ -> 0
