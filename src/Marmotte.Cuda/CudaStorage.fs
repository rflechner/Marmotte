namespace Marmotte.Cuda

/// Device pointers never escape the assembly. Metadata survives disposal, just as on the CPU.
type internal ICudaStorage =
    abstract Pointer: nativeint
    abstract Invalidate: unit -> unit
