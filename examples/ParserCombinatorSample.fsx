#load "../src/Marmotte.Core/Helpers.fs"
#load "../src/Marmotte.Core/FastStack.fs"
#load "../src/Marmotte.Core/ParserCombinator.fs"
#load "../src/Marmotte.Core/Jinja/JinjaSyntax.fs"
#load "../src/Marmotte.Core/Jinja/JinjaExpressionParser.fs"
#load "../src/Marmotte.Core/Jinja/JinjaParser.fs"

open Marmotte.Core

// Run from the repository root: dotnet fsi examples/ParserCombinatorSample.fsx
// Parsing builds syntax only. A future runtime will resolve variables from a context.
// See docs/jinja-parser-sample.md for supported syntax and parser composition.
let template = """{# Chat messages #}
{% for message in messages %}
{% if message.role == 'system' %}System: {{ message.content }}
{% elif message.role == 'user' and not hide_user %}User: {{ message.content }}
{% else %}{{ message.role }}: {{ message.content }}
{% endif %}
{% endfor %}"""

match JinjaParser.parse template with
| Ok ast -> printfn "Template AST:\n%A" ast
| Error error -> failwithf "Sample failed: %A" error

match JinjaParser.parse "hello\n{% if enabled %}missing closing tag" with
| Error error ->
    printfn "Invalid sample: line %d, column %d: %s"
        (error.Position.Line + 1) (error.Position.Column + 1) error.Message
| Ok _ -> failwith "Expected an unterminated conditional to fail"
