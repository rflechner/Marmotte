module Marmotte.Core.ChatPromptTests

open Marmotte.Core
open Xunit

[<Fact>]
let ``raw prompts bypass templates and remain byte for byte text`` () =
    let text = " {{ missing }}\r\n<bos> "
    for template in [None; Some "{% unsupported %}"] do
        Assert.Equal<Result<string, ChatPromptError>>(Ok text, ChatPrompt.render template Map.empty (ChatPrompt.Raw text))

[<Fact>]
let ``conversations supply messages and generation flag without inventing formatting`` () =
    let template = "{% for message in messages %}[{{ message.role }}]{{ message.content }};{% endfor %}{% if add_generation_prompt %}[assistant]{% endif %}"
    let context = Map.ofList ["messages", JinjaValue.None; "add_generation_prompt", JinjaValue.None]
    let messages = [{ Role = "custom"; Content = "{{ literal }}" }; { Role = "user"; Content = "Hello" }]
    for flag, suffix in [true, "[assistant]"; false, ""] do
        Assert.Equal<Result<string, ChatPromptError>>(
            Ok ("[custom]{{ literal }};[user]Hello;" + suffix),
            ChatPrompt.render (Some template) context (ChatPrompt.Conversation (messages, flag)))
    Assert.Equal<Result<string, ChatPromptError>>(
        Ok "", ChatPrompt.render (Some template) context (ChatPrompt.Conversation ([], false)))

[<Fact>]
let ``prompt errors distinguish missing syntax and runtime failures`` () =
    let prompt = ChatPrompt.Conversation ([], false)
    Assert.Equal(Error ChatPromptError.MissingTemplate, ChatPrompt.render None Map.empty prompt)
    match ChatPrompt.render (Some "{% set x = 1 %}") Map.empty prompt with
    | Error (ChatPromptError.Syntax _) -> ()
    | actual -> failwithf "Expected syntax error: %A" actual
    Assert.Equal(
        Error (ChatPromptError.Runtime (JinjaRuntimeError.MissingVariable ["bos_token"])),
        ChatPrompt.render (Some "{{ bos_token }}") Map.empty prompt)
    Assert.Equal<Result<string, ChatPromptError>>(Ok "", ChatPrompt.render (Some "") Map.empty prompt)
