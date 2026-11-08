module Marmotte.Core.JinjaParserTests

open System
open Marmotte.Core
open Xunit

let private parse input =
    match JinjaParser.parse input with
    | Ok nodes -> nodes
    | Error error -> failwithf "Unexpected parse error: %A" error

let private variable name = JinjaExpression.Variable [name]
let private integer value = JinjaExpression.Literal (JinjaLiteral.Integer value)
let private output expression = JinjaNode.Output expression

[<Fact>]
let ``empty templates and literal whitespace are preserved`` () =
    Assert.Equal<JinjaNode list>([], parse "")
    Assert.Equal<JinjaNode list>([JinjaNode.Text "a { b\r\n"], parse "a { b\r\n")
    Assert.Equal<JinjaNode list>(
        [output (JinjaExpression.Variable ["user"; "name"]); JinjaNode.Comment ""; JinjaNode.Text "\n"],
        parse "{{ user.name }}{##}\n")

[<Fact>]
let ``branches and loops retain order and whitespace`` () =
    let expected =
        [ JinjaNode.For ("message", variable "messages",
            [ JinjaNode.If (
                [ { Condition = variable "enabled"; Body = [output (variable "message")] }
                  { Condition = variable "fallback"; Body = [JinjaNode.Text " fallback "] } ],
                Some [JinjaNode.Text " none "]) ])
          JinjaNode.Text "\n" ]
    let input = "{% for message in messages %}{% if enabled %}{{ message }}{% elif fallback %} fallback {% else %} none {% endif %}{% endfor %}\n"
    Assert.Equal<JinjaNode list>(expected, parse input)

[<Fact>]
let ``else binds to the nearest conditional and absence differs from empty`` () =
    let inner = JinjaNode.If ([{ Condition = variable "b"; Body = [] }], Some [])
    Assert.Equal<JinjaNode list>(
        [JinjaNode.If ([{ Condition = variable "a"; Body = [inner] }], None)],
        parse "{% if a %}{% if b %}{% else %}{% endif %}{% endif %}")
    Assert.Equal<JinjaNode list>(
        [JinjaNode.If ([{ Condition = variable "a"; Body = [JinjaNode.For ("x", variable "xs", [])] }], None)],
        parse "{% if a %}{% for x in xs %}{% endfor %}{% endif %}")

[<Fact>]
let ``logical precedence is comparison then not then and then or`` () =
    let comparison = JinjaExpression.Compare (variable "b", [JinjaComparison.Equal, integer 2L])
    let expected = JinjaExpression.Or (variable "a", JinjaExpression.And (JinjaExpression.Not comparison, variable "c"))
    Assert.Equal<JinjaNode list>([output expected], parse "{{ a or not b == 2 and c }}")
    Assert.Equal<JinjaNode list>(
        [output (JinjaExpression.And (JinjaExpression.Or (variable "a", variable "b"), variable "c"))],
        parse "{{ (a or b) and c }}")

[<Fact>]
let ``comparison chains preserve operands and membership operators`` () =
    Assert.Equal<JinjaNode list>(
        [output (JinjaExpression.Compare (integer 0L,
            [JinjaComparison.LessThan, variable "count"; JinjaComparison.LessThanOrEqual, integer 10L]))],
        parse "{{ 0 < count <= 10 }}")
    Assert.Equal<JinjaNode list>(
        [output (JinjaExpression.Compare (variable "role", [JinjaComparison.NotIn, variable "roles"]))],
        parse "{{ role not in roles }}")

[<Fact>]
let ``comparison operators are distinct`` () =
    let cases =
        [ "==", JinjaComparison.Equal; "!=", JinjaComparison.NotEqual
          "<", JinjaComparison.LessThan; "<=", JinjaComparison.LessThanOrEqual
          ">", JinjaComparison.GreaterThan; ">=", JinjaComparison.GreaterThanOrEqual
          "in", JinjaComparison.In ]
    for operator, expected in cases do
        Assert.Equal<JinjaNode list>(
            [output (JinjaExpression.Compare (variable "a", [expected, variable "b"]))],
            parse ("{{ a " + operator + " b }}"))

[<Fact>]
let ``strings contain delimiters and decode supported escapes`` () =
    Assert.Equal<JinjaNode list>(
        [output (JinjaExpression.Literal (JinjaLiteral.String "}}{%\n\t\r\\\"'"))],
        parse "{{ '}}{%\\n\\t\\r\\\\\\\"\\\'' }}")
    Assert.Equal<JinjaNode list>(
        [JinjaNode.Comment " {{ ignored }} {% endif %} "],
        parse "{# {{ ignored }} {% endif %} #}")

[<Fact>]
let ``literal values and keyword prefixes are unambiguous`` () =
    let cases =
        [ "true", JinjaLiteral.Boolean true; "True", JinjaLiteral.Boolean true
          "false", JinjaLiteral.Boolean false; "False", JinjaLiteral.Boolean false
          "none", JinjaLiteral.None; "None", JinjaLiteral.None
          "-9223372036854775808", JinjaLiteral.Integer Int64.MinValue
          "9223372036854775807", JinjaLiteral.Integer Int64.MaxValue ]
    for source, literal in cases do
        Assert.Equal<JinjaNode list>([output (JinjaExpression.Literal literal)], parse ("{{ " + source + " }}"))
    for name in ["notable"; "trueValue"; "inside"; "order"; "android"] do
        Assert.Equal<JinjaNode list>([output (variable name)], parse ("{{ " + name + " }}"))

[<Theory>]
[<InlineData("{{")>]
[<InlineData("{{ }}")>]
[<InlineData("{{ user. }}")>]
[<InlineData("{# unfinished")>]
[<InlineData("{{ 'unfinished }}")>]
[<InlineData("{{ '\\q' }}")>]
[<InlineData("{{ 9223372036854775808 }}")>]
[<InlineData("{{ -9223372036854775809 }}")>]
[<InlineData("{{ a and }}")>]
[<InlineData("{{ a == }}")>]
[<InlineData("{{ a not }}")>]
[<InlineData("{{ (a }}")>]
[<InlineData("{{ a b }}")>]
[<InlineData("{{ name | upper }}")>]
[<InlineData("{{ user['name'] }}")>]
[<InlineData("{{ name() }}")>]
[<InlineData("{{ name is defined }}")>]
[<InlineData("{{- name }}")>]
[<InlineData("{% if %}{% endif %}")>]
[<InlineData("{% if a %}")>]
[<InlineData("{% if a %}{% endfor %}")>]
[<InlineData("{% if a %}{% else %}{% elif b %}{% endif %}")>]
[<InlineData("{% if a %}{% else %}{% else %}{% endif %}")>]
[<InlineData("{% if a %}{% endif extra %}")>]
[<InlineData("{% if a %}{% else extra %}{% endif %}")>]
[<InlineData("{% elif a %}")>]
[<InlineData("{% else %}")>]
[<InlineData("{% endif %}")>]
[<InlineData("{% endfor %}")>]
[<InlineData("{% for item inside items %}{% endfor %}")>]
[<InlineData("{% for true in items %}{% endfor %}")>]
[<InlineData("{% for item in items %}{% else %}{% endfor %}")>]
[<InlineData("{% set x = 1 %}")>]
let ``malformed or unsupported syntax returns an error`` input =
    match JinjaParser.parse input with
    | Error error -> Assert.NotEmpty error.Message
    | Ok ast -> failwithf "Expected error, got %A" ast

[<Fact>]
let ``errors locate the failure after multiline Unicode text`` () =
    match JinjaParser.parse "😀\r\n{{ }}" with
    | Error error ->
        Assert.Equal(7, error.Position.Offset)
        Assert.Equal(1, error.Position.Line)
        Assert.Equal(3, error.Position.Column)
    | Ok _ -> failwith "Expected error"

[<Fact>]
let ``resource limits reject excessive input without throwing`` () =
    let nested count = String.replicate count "{% if a %}" + String.replicate count "{% endif %}"
    parse (nested 64) |> ignore
    let invalid =
        [ nested 65
          "{{ " + String.replicate 65 "not " + "a }}"
          "{{ " + String.replicate 65 "(" + "a" + String.replicate 65 ")" + " }}"
          "{{ " + String.replicate 65 "a and " + "a }}"
          String.replicate 1_048_577 "x" ]
    for input in invalid do
        Assert.True(JinjaParser.parse input |> Result.isError)
    Assert.Throws<ArgumentNullException>(fun () -> JinjaParser.parse null |> ignore) |> ignore
