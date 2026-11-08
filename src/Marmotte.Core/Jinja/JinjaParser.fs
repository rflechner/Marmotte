/// Parsing for the documented Jinja subset; no evaluation or context lookup occurs here.
module Marmotte.Core.JinjaParser

open System
open Marmotte.Core
open Marmotte.Core.ParserCombinator
open Marmotte.Core.JinjaExpressionParser

let private statementName = symbol "{%" >>. token identifier
let private statement name = symbol "{%" >>. keyword name
// Closing delimiters must not consume whitespace belonging to subsequent literal text.
let private closeTag = pString "%}"

let private output = symbol "{{" >>. expression 0 .>> pString "}}" ||> JinjaNode.Output
let private comment =
    pString "{#" >>. pUntilStr "#}" .>> pString "#}"
    ||> fun content -> JinjaNode.Comment (content.ToString())

let private startsTag context =
    startsWith "{{" context || startsWith "{#" context || startsWith "{%" context

let private text : Parser<JinjaNode> =
    fun context ->
        let mutable length = 0
        let mutable rest = context
        while not rest.Remaining.IsEmpty && not (startsTag rest) do
            rest <- rest.Forward 1
            length <- length + 1
        Success (JinjaNode.Text (context.Remaining.Slice(0, length).ToString()), rest)

// A body stops before its own terminator. Other terminators are errors, so nested
// blocks cannot accidentally consume the closing tag belonging to their parent.
let rec private nodes depth terminators : Parser<JinjaNode list> =
    fun context ->
        let rec collect reversed context =
            if context.Remaining.IsEmpty then
                if List.isEmpty terminators then Success (List.rev reversed, context)
                else Failure ("Expected one of: " + String.concat ", " terminators, context)
            elif startsWith "{%" context then
                match run statementName context with
                | Failure (message, rest) -> Failure (message, rest)
                | Success (name, _) when List.contains name terminators -> Success (List.rev reversed, context)
                | Success (name, _) ->
                    match run (block depth name) context with
                    | Failure (message, rest) -> Failure (message, rest)
                    | Success (value, rest) -> collect (value :: reversed) rest
            else
                let parser =
                    if startsWith "{{" context then output
                    elif startsWith "{#" context then comment
                    else text
                match run parser context with
                | Failure (message, rest) -> Failure (message, rest)
                | Success (value, rest) -> collect (value :: reversed) rest
        collect [] context

and private block depth name : Parser<JinjaNode> =
    fun context ->
        if depth >= 64 then Failure ("Template nesting exceeds 64 levels", context)
        else
            match name with
            | "for" -> run (forBlock (depth + 1)) context
            | "if" -> run (ifBlock (depth + 1)) context
            | _ -> Failure ($"Unsupported or unexpected statement '{name}'", context)

and private forBlock depth =
    parsing {
        let! variable = statement "for" >>. variableName .>> pWhiteSpace .>> whitespace
        let! collection = keyword "in" >>. expression 0 .>> closeTag
        let! body = nodes depth ["endfor"]
        let! _ = statement "endfor" >>. closeTag
        return JinjaNode.For (variable, collection, body)
    }

and private ifBlock depth =
    let rec branches keywordName reversed =
        parsing {
            let! condition = statement keywordName >>. expression 0 .>> closeTag
            let! body = nodes depth ["elif"; "else"; "endif"]
            let accumulated = { Condition = condition; Body = body } :: reversed
            return! finish accumulated
        }
    and finish reversed : Parser<JinjaNode> =
        fun context ->
            match run statementName context with
            | Success ("elif", _) -> run (branches "elif" reversed) context
            | Success ("else", _) ->
                run (parsing {
                    let! _ = statement "else" >>. closeTag
                    let! body = nodes depth ["endif"]
                    let! _ = statement "endif" >>. closeTag
                    return JinjaNode.If (List.rev reversed, Some body)
                }) context
            | _ ->
                run (statement "endif" >>. closeTag
                     ||> fun _ -> JinjaNode.If (List.rev reversed, None)) context
    branches "if" []

/// Parse a complete template into syntax, without rendering or resolving variables.
/// Syntax errors and unsupported constructs return Error; null input raises ArgumentNullException.
/// Literal whitespace is preserved, including the final newline. See docs/jinja-parser-sample.md
/// for the supported grammar. Input is limited to 1,048,576 UTF-16 code units; block and
/// expression nesting are limited to 64 levels to bound allocation and recursive parsing.
let parse (input: string) : Result<JinjaNode list, JinjaParseError> =
    ArgumentNullException.ThrowIfNull input
    if input.Length > 1_048_576 then
        Error { Message = "Template exceeds 1,048,576 UTF-16 code units"; Position = Position.Zero }
    else
        match run (nodes 0 []) (ParsingContext.FromString input) with
        | Success (ast, _) -> Ok ast
        | Failure (message, context) -> Error { Message = message; Position = context.Position }
