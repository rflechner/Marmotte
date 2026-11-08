namespace Marmotte.Core

/// A literal value in the supported Jinja syntax. Integers are signed 64-bit values.
[<RequireQualifiedAccess>]
type JinjaLiteral =
    | String of string
    | Integer of int64
    | Boolean of bool
    | None

/// A comparison or membership operation; evaluation belongs to a runtime.
[<RequireQualifiedAccess>]
type JinjaComparison =
    | Equal
    | NotEqual
    | LessThan
    | LessThanOrEqual
    | GreaterThan
    | GreaterThanOrEqual
    | In
    | NotIn

/// Unevaluated expressions, independent of any variable context.
/// Comparison chains retain each operand once: a < b < c is not (a < b) < c.
[<RequireQualifiedAccess>]
type JinjaExpression =
    | Variable of path: string list
    | Literal of JinjaLiteral
    | Not of JinjaExpression
    | And of left: JinjaExpression * right: JinjaExpression
    | Or of left: JinjaExpression * right: JinjaExpression
    | Compare of first: JinjaExpression * rest: (JinjaComparison * JinjaExpression) list

/// Template syntax in source order. Text and comments retain their original whitespace.
/// An absent else branch differs from an explicitly empty else branch.
[<RequireQualifiedAccess>]
type JinjaNode =
    | Text of string
    | Output of JinjaExpression
    | Comment of string
    | For of variable: string * collection: JinjaExpression * body: JinjaNode list
    | If of branches: JinjaBranch list * elseBody: JinjaNode list option

/// One if or elif branch, kept in source order for short-circuit evaluation.
and JinjaBranch =
    { Condition: JinjaExpression
      Body: JinjaNode list }

/// A syntax error. Positions are zero-based UTF-16 offsets and columns;
/// LF advances the line, including within CRLF. CR alone advances the column.
type JinjaParseError =
    { Message: string
      Position: ParserCombinator.Position }
