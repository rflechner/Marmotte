#load "SyntheticModel.fsx"

open System
open Marmotte.Core
open Marmotte.Cpu
open SyntheticModel

// Use --sample for probabilistic selection; greedy remains the default. Keep one sampler
// for the whole sequence so the random state advances instead of restarting for every token.
let stochastic = fsi.CommandLineArgs |> Array.contains "--sample"
let parameters = TokenSamplingParameters.Create(temperature = 0.8, topK = Some 3, topP = 0.9)
let sampler = TokenSampler(parameters, seed = 42)
let chooseToken (logits: ReadOnlyMemory<single>) =
    if stochastic then sampler.Sample(logits.Span)
    else TokenSampling.greedy logits.Span

let run () =
    let source, layout = createExample ()
    use source = source
    use backend = new CpuBackend() :> ITensorBackend
    use model = TransformerModel.Load(backend, source, layout)
    use sequence = GenerationSession.Create(model, 8, 2, TensorEncoding.F32, TensorEncoding.F32)
    // --eos treats ID 2 as an invented end token for this synthetic model.
    let endIds = if fsi.CommandLineArgs |> Array.contains "--eos" then [| 2 |] else Array.empty
    let generator = TokenGenerator(sequence, 3, ReadOnlyMemory<int>(endIds), chooseToken)
    let mutable emitted = 0
    let onToken token =
        emitted <- emitted + 1
        printfn "Generated token %d: %d (cached: %d)" emitted token sequence.ProcessedTokenCount
    printfn "Prompt: 3 tokens (batch size %d)" sequence.BatchSize
    let result = generator.Generate(ReadOnlyMemory<int>([| 0; 1; 2 |]), onToken = onToken)
    printfn "Stopped: %A; returned: %d; cached: %d" result.StopReason result.Tokens.Length sequence.ProcessedTokenCount

run ()
