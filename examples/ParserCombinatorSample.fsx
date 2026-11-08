#load "../src/Marmotte.Core/Helpers.fs"
#load "../src/Marmotte.Core/FastStack.fs"
#load "../src/Marmotte.Core/ParserCombinator.fs"

open System
open Marmotte.Core.ParserCombinator

// Run from the repository root: dotnet fsi examples/ParserCombinatorSample.fsx
// This teaching subset builds an AST; it does not evaluate or render templates.
// See docs/jinja-parser-sample.md for the grammar and deliberate limitations.
type TemplateNode =
    | Text of string
    | Output of path: string list
    | Comment of string
    | For of variable: string * collection: string list * body: TemplateNode list

type TemplateError =
    { Message: string
      Position: Position }

// Step 1: compose characters into identifiers, then identifiers into paths.
let isIdentifierStart character =
    Char.IsAsciiLetter character || character = '_'

let isIdentifierPart character =
    isIdentifierStart character || Char.IsAsciiDigit character

let identifier =
    satisfy isIdentifierStart .>>. zeroOrMany (satisfy isIdentifierPart)
    ||> fun (first, rest) -> string first + rest.ToString()

let whitespace = pSkip pWhiteSpace
let requiredWhitespace = pWhiteSpace >>. whitespace
let token parser = parser .>> whitespace

// `many` requires at least one match in this library; the alternative permits none.
let succeed value : Parser<_> = fun context -> Success (value, context)
let path =
    token identifier .>>. (many (token (pChar '.') >>. token identifier) <|> succeed [])
    ||> fun (first, rest) -> first :: rest

// Step 2: delimiters surround a composed parser. Closing delimiters are mandatory.
let output = pString "{{" >>. whitespace >>. path .>> pString "}}" ||> Output
let comment =
    pString "{#" >>. pUntilStr "#}" .>> pString "#}"
    ||> fun content -> Comment (content.ToString())

let forHeader =
    pString "{%" >>. whitespace >>. pString "for" >>. requiredWhitespace
    >>. (identifier .>> requiredWhitespace)
    .>>. (pString "in" >>. requiredWhitespace >>. path)
    .>> pString "%}"

let endFor =
    pString "{%" >>. whitespace >>. token (pString "endfor") .>> pString "%}"

let startsWith (prefix: string) (context: ParsingContext) =
    context.Remaining.Span.StartsWith(prefix.AsSpan(), StringComparison.Ordinal)

let startsTag context =
    [ "{{"; "{#"; "{%" ] |> List.exists (fun prefix -> startsWith prefix context)

let text : Parser<TemplateNode> =
    fun context ->
        let mutable length = 0
        let mutable rest = context

        while not rest.Remaining.IsEmpty && not (startsTag rest) do
            rest <- rest.Forward 1
            length <- length + 1

        if length = 0 then
            Failure ("Expected literal text", context)
        else
            Success (Text (context.Remaining.Slice(0, length).ToString()), rest)

// Step 3: recurse only for block bodies. Dispatch after an opening delimiter so a
// malformed tag cannot backtrack into literal text or silently truncate the AST.
let rec nodes depth insideLoop : Parser<TemplateNode list> =
    fun context ->
        let rec collect reversed context =
            if context.Remaining.IsEmpty then
                if insideLoop then Failure ("Expected {% endfor %}", context)
                else Success (List.rev reversed, context)
            else
                match run endFor context with
                | Success (_, rest) when insideLoop -> Success (List.rev reversed, rest)
                | Success _ -> Failure ("Unexpected {% endfor %}", context)
                | Failure _ ->
                    match run (node depth) context with
                    | Failure (message, failedContext) -> Failure (message, failedContext)
                    | Success (value, rest) -> collect (value :: reversed) rest

        collect [] context

and node depth : Parser<TemplateNode> =
    fun context ->
        if startsWith "{{" context then run output context
        elif startsWith "{#" context then run comment context
        elif startsWith "{%" context then
            if depth >= 64 then Failure ("At most 64 nested loops are supported", context)
            else
                let loop =
                    forHeader .>>. nodes (depth + 1) true
                    ||> fun ((variable, collection), body) -> For (variable, collection, body)

                run loop context
        else run text context

/// Parse the complete teaching subset, preserving all literal whitespace.
/// Error positions use zero-based UTF-16 offsets, lines and columns.
let parseTemplate input =
    match run (nodes 0 false) (ParsingContext.FromString input) with
    | Success (ast, _) -> Ok ast
    | Failure (message, context) -> Error { Message = message; Position = context.Position }

// Step 4: inspect a small chat-shaped template without requiring a model file.
let template =
    "{# Chat messages #}\n{% for message in messages %}{{ message.role }}: {{ message.content }}\n{% endfor %}"

match parseTemplate template with
| Ok ast -> printfn "Template AST:\n%A" ast
| Error error -> failwithf "Sample failed: %A" error

// Small executable checks cover order, nesting, empty input and rejected syntax.
let expectAst input expected =
    match parseTemplate input with
    | Ok actual when actual = expected -> ()
    | actual -> failwithf "Unexpected parse result for %A: %A" input actual

expectAst "" []
expectAst "a { b\n" [ Text "a { b\n" ]
expectAst "{{ user.name }}{##}{{ other }}"
    [ Output [ "user"; "name" ]; Comment ""; Output [ "other" ] ]
expectAst "{% for group in groups %}{% for item in group.items %}{{ item }}{% endfor %}{% endfor %}"
    [ For ("group", [ "groups" ], [ For ("item", [ "group"; "items" ], [ Output [ "item" ] ]) ]) ]
expectAst "{% for item in items %}{% endfor %}" [ For ("item", [ "items" ], []) ]

let invalidTemplates =
    [ "{{ name"
      "{{ }}"
      "{{ user. }}"
      "{# unfinished"
      "{% for item in items %}unfinished"
      "{% endfor %}"
      "{% if enabled %}text{% endif %}"
      "{{ name | upper }}"
      "{{- name }}"
      "{% for item inside items %}{% endfor %}"
      String.replicate 65 "{% for item in items %}" + String.replicate 65 "{% endfor %}" ]

for input in invalidTemplates do
    match parseTemplate input with
    | Error _ -> ()
    | Ok ast -> failwithf "Expected rejection for %A, got %A" input ast

match parseTemplate "hello\n{{ }}" with
| Error error when error.Position.Line = 1 && error.Position.Column = 3 && error.Position.Offset = 9 ->
    printfn "Invalid sample: line %d, column %d: %s"
        (error.Position.Line + 1) (error.Position.Column + 1) error.Message
| actual -> failwithf "Unexpected error location: %A" actual

printfn "All sample checks passed."
