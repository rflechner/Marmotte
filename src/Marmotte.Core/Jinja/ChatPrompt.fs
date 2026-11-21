namespace Marmotte.Core

open System

/// One text-only chat message. Role names are template data, not a fixed role enum.
/// Content is inserted verbatim and never interpreted as template syntax.
type ChatMessage =
    { Role: string
      Content: string }

/// Explicit choice between an unchanged raw prompt and a templated conversation.
[<RequireQualifiedAccess>]
type ChatPrompt =
    | Raw of string
    | Conversation of messages: ChatMessage list * addGenerationPrompt: bool

/// Prompt formatting failures distinguish missing configuration, syntax and evaluation.
[<RequireQualifiedAccess>]
type ChatPromptError =
    | MissingTemplate
    | Syntax of JinjaParseError
    | Runtime of JinjaRuntimeError

[<RequireQualifiedAccess>]
module ChatPrompt =
    /// Return raw text unchanged, or render a conversation with the supplied template.
    /// The context supplies optional template variables such as bos_token and eos_token;
    /// messages and add_generation_prompt always come from the Conversation argument.
    /// No roles, separators, boundaries or assistant prefix are invented. No tokenization
    /// occurs. Unsupported templates return Syntax, even in an unreachable branch.
    /// Null inputs/message fields raise ArgumentNullException. Raw ignores template syntax
    /// and rendering budgets; conversations use JinjaRuntime's default limits.
    let render (template: string option) (context: Map<string, JinjaValue>) (prompt: ChatPrompt)
        : Result<string, ChatPromptError> =
        ArgumentNullException.ThrowIfNull context
        ArgumentNullException.ThrowIfNull prompt
        match prompt with
        | ChatPrompt.Raw text ->
            ArgumentNullException.ThrowIfNull text
            Ok text
        | ChatPrompt.Conversation (messages, addGenerationPrompt) ->
            ArgumentNullException.ThrowIfNull messages
            let messageValues =
                messages |> List.map (fun message ->
                    ArgumentNullException.ThrowIfNull message
                    ArgumentNullException.ThrowIfNull message.Role
                    ArgumentNullException.ThrowIfNull message.Content
                    JinjaValue.Object (Map.ofList
                        [ "role", JinjaValue.String message.Role
                          "content", JinjaValue.String message.Content ]))
            match template with
            | None -> Error ChatPromptError.MissingTemplate
            | Some source ->
                JinjaParser.parse source
                |> Result.mapError ChatPromptError.Syntax
                |> Result.bind (fun nodes ->
                    let variables =
                        context
                        |> Map.add "messages" (JinjaValue.List messageValues)
                        |> Map.add "add_generation_prompt" (JinjaValue.Boolean addGenerationPrompt)
                    JinjaRuntime.render variables nodes |> Result.mapError ChatPromptError.Runtime)
