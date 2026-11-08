#load "SyntheticModel.fsx"

open System
open Marmotte.Core
open Marmotte.Cpu
open SyntheticModel

let run () =
    let source, layout = createExample ()
    use source = source
    use backend = new CpuBackend() :> ITensorBackend
    use model = TransformerModel.Load(backend, source, layout)
    use cache = backend.AllocateKeyValueCache(KeyValueCacheLayout.Create(1, 8, 1, 2, TensorEncoding.F32, TensorEncoding.F32))
    use logits = model.Forward(cache, 0, ReadOnlyMemory<int>([| 1; 2 |]))
    let values = Array.zeroCreate<single> logits.Shape.ElementCount
    backend.Download(logits, values.AsMemory())
    for token = 0 to 1 do
        let row = values[token * 4 .. token * 4 + 3] |> Array.map (sprintf "%.4f") |> String.concat "; "
        printfn "Token %d logits: [%s]" token row
    printfn "Cached positions: %d" (cache.StoredPositions 0)

run ()
