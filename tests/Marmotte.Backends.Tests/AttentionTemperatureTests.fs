module Marmotte.Backends.AttentionTemperatureTests

open System
open Marmotte.Core
open Marmotte.Cpu
open Xunit

let private cpu () = new CpuBackend() :> ITensorBackend

let private download (backend: ITensorBackend) (tensor: ITensor) =
    let values = Array.zeroCreate<single> tensor.Shape.ElementCount
    backend.Download(tensor, values.AsMemory())
    values

/// The target model's parameters: attention.temperature_scale 0.1 and an original context of 16,384.
let private ministral = AttentionTemperature.Create(float 0.1f, 16384)

let private apply (backend: ITensorBackend) (dimensions: int array) (values: single array) (positions: int array) temperature =
    use queries = backend.Upload(TensorShape.Create dimensions, ReadOnlyMemory<single>(values))
    use destination = backend.Allocate(TensorShape.Create dimensions)
    backend.ApplyAttentionTemperature(queries, ReadOnlyMemory<int>(positions), temperature, destination)
    Assert.Equal<single>(values, download backend queries)
    download backend destination

let private randomValues (random: Random) count =
    Array.init count (fun _ -> float32 (random.NextDouble() * 2.0 - 1.0))

[<Fact>]
let ``positions within the original context leave queries unchanged`` () =
    use backend = cpu ()
    let values = randomValues (Random(3031)) (3 * 4096)

    Assert.Equal<single>(values, apply backend [| 3; 4096 |] values [| 0; 1; 16383 |] ministral)

[<Fact>]
let ``each token is multiplied by the factor of its own position`` () =
    use backend = cpu ()
    let positions = [| 0; 16384; 40000; 262143 |]
    let values = randomValues (Random(3233)) (4 * 4096)
    // Every value of token t is multiplied once by its F32 factor, in vector lanes and the tail.
    let expected = values |> Array.mapi (fun index value -> value * ministral.Factor(positions[index / 4096]))

    Assert.Equal<single>(expected, apply backend [| 4; 4096 |] values positions ministral)
    Assert.Equal<single>(expected, apply backend [| 4; 32; 128 |] values positions ministral)

    // A value of 1 at position 16,384 becomes 1 + 0.1 ln 2.
    let scaled = apply backend [| 1; 7 |] (Array.create 7 1.0f) [| 16384 |] ministral
    Assert.All(scaled, fun value -> Assert.Equal(1.06931, float value, 5))

[<Fact>]
let ``attention temperature supports in-place scaling and empty batches`` () =
    use backend = cpu ()
    let values = randomValues (Random(3435)) 16
    let separate = apply backend [| 2; 8 |] values [| 20000; 50000 |] ministral
    use queries = backend.Upload(TensorShape.Create [| 2; 8 |], ReadOnlyMemory<single>(values))

    backend.ApplyAttentionTemperature(queries, ReadOnlyMemory<int>([| 20000; 50000 |]), ministral, queries)

    Assert.Equal<single>(separate, download backend queries)
    Assert.Empty(apply backend [| 0; 128 |] [||] [||] ministral)

[<Fact>]
let ``attention temperature validates shapes positions ownership and disposal before writing`` () =
    use backend = cpu ()
    use otherBackend = cpu ()
    let shape = TensorShape.Create [| 2; 8 |]
    use queries = backend.Upload(shape, ReadOnlyMemory<single>(Array.create 16 1.0f))
    use destination = backend.Upload(shape, ReadOnlyMemory<single>(Array.create 16 42.0f))
    use vector = backend.Allocate(TensorShape.Create [| 8 |])
    use otherShape = backend.Allocate(TensorShape.Create [| 2; 4 |])
    use foreign = otherBackend.Allocate(shape)
    let disposed = backend.Allocate(shape)
    disposed.Dispose()
    let positions = ReadOnlyMemory<int>([| 0; 20000 |])

    let tokenError = Assert.Throws<ArgumentException>(fun () -> backend.ApplyAttentionTemperature(queries, ReadOnlyMemory<int>([| 0 |]), ministral, destination))
    Assert.Contains("1 tokens", tokenError.Message)
    Assert.Throws<ArgumentException>(fun () -> backend.ApplyAttentionTemperature(vector, ReadOnlyMemory<int>(Array.zeroCreate 8), ministral, vector)) |> ignore
    Assert.Throws<ArgumentException>(fun () -> backend.ApplyAttentionTemperature(queries, positions, ministral, otherShape)) |> ignore
    let positionError = Assert.Throws<ArgumentOutOfRangeException>(fun () -> backend.ApplyAttentionTemperature(queries, ReadOnlyMemory<int>([| 20000; -1 |]), ministral, destination))
    Assert.Contains("index 1", positionError.Message)
    Assert.Throws<ArgumentException>(fun () -> backend.ApplyAttentionTemperature(foreign, positions, ministral, destination)) |> ignore
    Assert.Throws<ObjectDisposedException>(fun () -> backend.ApplyAttentionTemperature(disposed, positions, ministral, destination)) |> ignore
    Assert.Throws<ArgumentNullException>(fun () -> backend.ApplyAttentionTemperature(queries, positions, Unchecked.defaultof<AttentionTemperature>, destination)) |> ignore
    Assert.Throws<ArgumentNullException>(fun () -> backend.ApplyAttentionTemperature(Unchecked.defaultof<ITensor>, positions, ministral, destination)) |> ignore
    Assert.Equal<single>(Array.create 16 42.0f, download backend destination)
