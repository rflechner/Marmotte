#load "SyntheticModel.fsx"

open System
open Marmotte.Core
open Marmotte.Cpu
open SyntheticModel

// The caller selects the library's greedy policy. Invalid model scores are reported explicitly;
// probabilistic sampling and EOS/length policy remain separate features.
let chooseToken (logits: single array) =
    match TokenSampling.greedy (ReadOnlySpan<single>(logits)) with
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
