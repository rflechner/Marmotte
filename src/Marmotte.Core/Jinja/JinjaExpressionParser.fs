module internal Marmotte.Core.JinjaExpressionParser

open System
open System.Globalization
open System.Text
open Marmotte.Core
open Marmotte.Core.ParserCombinator

// Bind keeps failures at the point of failure rather than trying another grammar.
type ParserBuilder() =
    member _.Bind(parser, continuation) : Parser<_> =
        fun context ->
            match run parser context with
            | Success (value, rest) -> run (continuation value) rest
            | Failure (message, rest) -> Failure (message, rest)
    member _.Return(value) : Parser<_> = fun context -> Success (value, context)
    member _.ReturnFrom(parser: Parser<_>) = parser
    member _.Delay(factory) : Parser<_> = fun context -> run (factory ()) context

let parsing = ParserBuilder()
let fail message : Parser<_> = fun context -> Failure (message, context)
let whitespace = pSkip pWhiteSpace
let token parser = parser .>> whitespace
let symbol value = token (pString value)
let startsWith (value: string) (context: ParsingContext) =
    context.Remaining.Span.StartsWith(value.AsSpan(), StringComparison.Ordinal)

let private isIdentifierStart character = Char.IsAsciiLetter character || character = '_'
let private isIdentifierPart character = isIdentifierStart character || Char.IsAsciiDigit character

let identifier =
    satisfy isIdentifierStart .>>. zeroOrMany (satisfy isIdentifierPart)
    ||> fun (first, rest) -> string first + rest.ToString()

let keyword value : Parser<string> =
    fun context ->
        match run (pString value) context with
        | Success (_, rest) when rest.Remaining.IsEmpty || not (isIdentifierPart rest.Remaining.Span[0]) ->
            run (whitespace ||> fun () -> value) rest
        | _ -> Failure ($"Expected keyword '{value}'", context)

let private reserved =
    set [ "and"; "or"; "not"; "in"; "is"; "if"; "else"; "elif"; "for";
          "true"; "false"; "none"; "True"; "False"; "None" ]

let variableName =
    parsing {
        let! name = identifier
        if Set.contains name reserved then return! fail "Expected a variable name, not a keyword"
        else return name
    }

let private stringLiteral : Parser<JinjaExpression> =
    fun context ->
        let quote = context.Remaining.Span[0]
        let content = StringBuilder()
        let mutable rest = context.Forward 1
        let mutable closed = false
        let mutable error = None

        while not closed && error.IsNone && not rest.Remaining.IsEmpty do
            let character = rest.Remaining.Span[0]
            if character = quote then
                closed <- true
                rest <- rest.Forward 1
            elif character = '\\' then
                let escapePosition = rest
                rest <- rest.Forward 1
                if rest.Remaining.IsEmpty then
                    error <- Some ("Unterminated string escape", escapePosition)
                else
                    let decoded =
                        match rest.Remaining.Span[0] with
                        | '\\' -> Some '\\'
                        | '\'' -> Some '\''
                        | '"' -> Some '"'
                        | 'n' -> Some '\n'
                        | 'r' -> Some '\r'
                        | 't' -> Some '\t'
                        | _ -> None
                    match decoded with
                    | Some value ->
                        content.Append(value) |> ignore
                        rest <- rest.Forward 1
                    | None -> error <- Some ("Unsupported string escape", escapePosition)
            else
                content.Append(character) |> ignore
                rest <- rest.Forward 1

        match error with
        | Some (message, failedContext) -> Failure (message, failedContext)
        | None when not closed -> Failure ("Expected closing string quote", rest)
        | None -> Success (JinjaExpression.Literal (JinjaLiteral.String (content.ToString())), rest)

let private integerLiteral : Parser<JinjaExpression> =
    fun context ->
        let digits = (zeroOrOne (pChar '-') .>>. manySatisfy Char.IsAsciiDigit)
        match run digits context with
        | Failure (message, rest) -> Failure (message, rest)
        | Success (_, rest) ->
            let length = rest.Position.Offset - context.Position.Offset
            let source = context.Remaining.Slice(0, length).ToString()
            match Int64.TryParse(source, NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture) with
            | true, value -> Success (JinjaExpression.Literal (JinjaLiteral.Integer value), rest)
            | _ -> Failure ("Integer literal is outside the signed 64-bit range", context)

let private variable =
    parsing {
        let! first = token variableName
        let rec members reversed : Parser<JinjaExpression> =
            fun context ->
                if startsWith "." context then
                    run (parsing {
                        let! name = symbol "." >>. token identifier
                        return! members (name :: reversed)
                    }) context
                else Success (JinjaExpression.Variable (List.rev reversed), context)
        return! members [first]
    }

let private comparisonOperator =
    choice [ symbol "==" ||> fun _ -> JinjaComparison.Equal
             symbol "!=" ||> fun _ -> JinjaComparison.NotEqual
             symbol "<=" ||> fun _ -> JinjaComparison.LessThanOrEqual
             symbol ">=" ||> fun _ -> JinjaComparison.GreaterThanOrEqual
             symbol "<" ||> fun _ -> JinjaComparison.LessThan
             symbol ">" ||> fun _ -> JinjaComparison.GreaterThan
             keyword "in" ||> fun _ -> JinjaComparison.In
             keyword "not" >>. keyword "in" ||> fun _ -> JinjaComparison.NotIn ]

// Limit recursive expression shape as well as template nesting, before descending.
let private guard depth parser : Parser<_> =
    fun context ->
        if depth >= 64 then Failure ("Expression nesting exceeds 64 levels", context)
        else run parser context

let rec expression depth : Parser<JinjaExpression> =
    guard depth (logical depth "or" JinjaExpression.Or conjunction)

and private conjunction depth : Parser<JinjaExpression> =
    logical depth "and" JinjaExpression.And negation

and private logical depth operator construct operand : Parser<JinjaExpression> =
    parsing {
        let! first = operand depth
        let rec rest count left : Parser<JinjaExpression> =
            fun context ->
                match run (keyword operator) context with
                | Failure _ -> Success (left, context)
                | Success (_, next) ->
                    run (guard count (parsing {
                        let! right = operand (count + 1)
                        return! rest (count + 1) (construct (left, right))
                    })) next
        return! rest depth first
    }

and private negation depth : Parser<JinjaExpression> =
    guard depth (fun context ->
        match run (keyword "not") context with
        | Success (_, rest) -> run (negation (depth + 1) ||> JinjaExpression.Not) rest
        | Failure _ -> run (comparison depth) context)

and private comparison depth : Parser<JinjaExpression> =
    parsing {
        let! first = atom depth
        let rec comparisons reversed : Parser<JinjaExpression> =
            fun context ->
                match run comparisonOperator context with
                | Failure _ ->
                    let result =
                        if List.isEmpty reversed then first
                        else JinjaExpression.Compare (first, List.rev reversed)
                    Success (result, context)
                | Success (operator, rest) ->
                    run (parsing {
                        let! right = atom depth
                        return! comparisons ((operator, right) :: reversed)
                    }) rest
        return! comparisons []
    }

and private atom depth : Parser<JinjaExpression> =
    guard depth (fun context ->
        if context.Remaining.IsEmpty then Failure ("Expected an expression", context)
        else
            let parser =
                match context.Remaining.Span[0] with
                | '(' -> symbol "(" >>. expression (depth + 1) .>> symbol ")"
                | '\'' | '"' -> token stringLiteral
                | '-' -> token integerLiteral
                | character when Char.IsAsciiDigit character -> token integerLiteral
                | _ ->
                    choice [ keyword "true" <|> keyword "True" ||> fun _ -> JinjaExpression.Literal (JinjaLiteral.Boolean true)
                             keyword "false" <|> keyword "False" ||> fun _ -> JinjaExpression.Literal (JinjaLiteral.Boolean false)
                             keyword "none" <|> keyword "None" ||> fun _ -> JinjaExpression.Literal JinjaLiteral.None
                             variable ]
            run parser context)
