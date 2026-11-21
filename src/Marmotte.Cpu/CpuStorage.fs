namespace Marmotte.Cpu

/// Storage that its backend releases when the backend itself is disposed.
type internal ICpuStorage =
    /// Marks the storage disposed and drops its buffer, without notifying the backend.
    abstract Release: unit -> unit
