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
let chooseToken (logits: single array) =
    let result =
        if stochastic then sampler.Sample(ReadOnlySpan<single>(logits))
        else TokenSampling.greedy (ReadOnlySpan<single>(logits))
    match result with
    | Ok token -> token
    | Error error -> invalidOp $"Token selection failed: {error}."

let run () =
    let source, layout = createExample ()
    use source = source
    use backend = new CpuBackend() :> ITensorBackend
    use model = TransformerModel.Load(backend, source, layout)
    use sequence = GenerationSession.Create(model, 8, 2, TensorEncoding.F32, TensorEncoding.F32)
    let mutable logits = sequence.Prefill(ReadOnlyMemory<int>([| 0; 1; 2 |]))
    printfn "Prompt cached: %d tokens (batch size %d)" sequence.ProcessedTokenCount sequence.BatchSize
    // Displaying the selected ID does not put it in the cache. Decode consumes it only when
    // we need the following token's logits. This avoids an unnecessary final forward call.
    for step = 1 to 3 do
        let token = chooseToken logits
        printfn "Generated token %d: %d (cached: %d)" step token sequence.ProcessedTokenCount
        if step < 3 then logits <- sequence.Decode(token)

run ()
