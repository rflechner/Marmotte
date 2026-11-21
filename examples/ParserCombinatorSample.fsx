#load "../src/Marmotte.Core/Helpers.fs"
#load "../src/Marmotte.Core/FastStack.fs"
#load "../src/Marmotte.Core/ParserCombinator.fs"
#load "../src/Marmotte.Core/Jinja/JinjaSyntax.fs"
#load "../src/Marmotte.Core/Jinja/JinjaExpressionParser.fs"
#load "../src/Marmotte.Core/Jinja/JinjaParser.fs"
#load "../src/Marmotte.Core/Jinja/JinjaValue.fs"
#load "../src/Marmotte.Core/Jinja/JinjaRuntime.fs"
#load "../src/Marmotte.Core/Jinja/ChatPrompt.fs"

open Marmotte.Core

// Run from the repository root: dotnet fsi examples/ParserCombinatorSample.fsx
// Parsing builds syntax; rendering resolves variables from an explicit context.
// See docs/jinja-parser-sample.md for supported syntax and parser composition.
let template = """{# Chat messages #}
{% for message in messages %}
{% if message.role == 'system' %}System: {{ message.content }}
{% elif message.role == 'user' and not hide_user_label %}User: {{ message.content }}
{% else %}{{ message.role }}: {{ message.content }}
{% endif %}
{% endfor %}"""

let ast =
    match JinjaParser.parse template with
    | Ok nodes -> nodes
    | Error error -> failwithf "Sample failed: %A" error

printfn "Template AST:\n%A" ast

match JinjaParser.parse "hello\n{% if enabled %}missing closing tag" with
| Error error ->
    printfn "Invalid sample: line %d, column %d: %s"
        (error.Position.Line + 1) (error.Position.Column + 1) error.Message
| Ok _ -> failwith "Expected an unterminated conditional to fail"

// Reuse the parsed template with a context containing ordinary immutable data.
let message role content =
    JinjaValue.Object (Map.ofList
        [ "role", JinjaValue.String role
          "content", JinjaValue.String content ])

let context =
    Map.ofList
        [ "messages", JinjaValue.List
            [ message "system" "Answer briefly."
              message "user" "Hello!"
              message "assistant" "Welcome." ]
          "hide_user_label", JinjaValue.Boolean false ]

match JinjaRuntime.render context ast with
| Ok text ->
    let expected = "\n\nSystem: Answer briefly.\n\n\nUser: Hello!\n\n\nassistant: Welcome.\n\n"
    if text <> expected then failwithf "Unexpected rendered prompt: %A" text
    printfn "Rendered template:\n%s" text
| Error error -> failwithf "Cannot render sample: %A" error

// The same AST can be rendered again with another context; unused branches stay unevaluated.
let fallbackContext = context |> Map.add "hide_user_label" (JinjaValue.Boolean true)
match JinjaRuntime.render fallbackContext ast with
| Ok text -> printfn "Rendered with hide_user_label=true:\n%s" text
| Error error -> failwithf "Cannot render second context: %A" error

// The prompt API constructs messages/add_generation_prompt for callers.
let chatTemplate =
    "{{ bos_token }}{% for message in messages %}[{{ message.role }}]{{ message.content }}{{ eos_token }}{% endfor %}{% if add_generation_prompt %}[assistant]{% endif %}"
let boundaries =
    Map.ofList [
        "bos_token", JinjaValue.String "<bos>"
        "eos_token", JinjaValue.String "<eos>"
    ]

let chatHistory = [
    { Role = "user"; Content = "Hello!" }
    { Role = "assistant"; Content = "Hello, how can I help you ?" }
    { Role = "user"; Content = "What time is it ?" }
]
let prompt =
    ChatPrompt.Conversation (chatHistory, true)

match ChatPrompt.render (Some chatTemplate) boundaries prompt with
| Ok text ->
    if text <> "<bos>[user]Hello!<eos>[assistant]" then failwithf "Unexpected chat prompt: %A" text
    printfn "Chat prompt: %s" text
| Error error -> failwithf "Cannot format chat prompt: %A" error
