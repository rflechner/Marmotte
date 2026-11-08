namespace Marmotte.Core

/// Immutable runtime data. Objects expose map keys only, never reflected .NET members.
/// Strings are literal data and are never parsed as template source.
[<RequireQualifiedAccess>]
type JinjaValue =
    | String of string
    | Integer of int64
    | Boolean of bool
    | None
    | List of JinjaValue list
    | Object of Map<string, JinjaValue>

/// Expected evaluation failures. Runtime errors have no source span because the AST
/// does not retain spans. Missing paths identify the requested variable or member.
[<RequireQualifiedAccess>]
type JinjaRuntimeError =
    | MissingVariable of path: string list
    | InvalidOperation of message: string
    | LimitExceeded of message: string

/// Per-render budgets. Output is measured in UTF-16 code units; steps include nodes,
/// expressions, loop iterations, member lookups and value comparisons.
/// Positive values are required. Depth cannot exceed 128, including caller-built ASTs.
type JinjaRenderLimits =
    { MaxOutputLength: int
      MaxSteps: int
      MaxDepth: int }
    /// Bounded defaults for prompt rendering, independent of model size.
    static member Default =
        { MaxOutputLength = 1_048_576
          MaxSteps = 100_000
          MaxDepth = 128 }
