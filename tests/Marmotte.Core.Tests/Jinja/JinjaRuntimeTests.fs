module Marmotte.Core.JinjaRuntimeTests

open System
open System.Globalization
open Marmotte.Core
open Xunit

let private parse source =
    match JinjaParser.parse source with
    | Ok nodes -> nodes
    | Error error -> failwithf "Invalid fixture: %A" error

let private render context source = JinjaRuntime.render context (parse source)
let private assertText expected actual = Assert.Equal<Result<string, JinjaRuntimeError>>(Ok expected, actual)

[<Fact>]
let ``context strings are literal data and whitespace is preserved`` () =
    let context = Map.ofList ["user", JinjaValue.Object (Map.ofList ["name", JinjaValue.String "<b>{{ missing }}</b>"])]
    render context " {# ignored #}{{ user.name }}\r\n" |> assertText " <b>{{ missing }}</b>\r\n"
    render Map.empty "" |> assertText ""

[<Fact>]
let ``truthiness covers every value category`` () =
    let cases =
        [ JinjaValue.None, false
          JinjaValue.Boolean false, false
          JinjaValue.Boolean true, true
          JinjaValue.Integer 0L, false
          JinjaValue.Integer -1L, true
          JinjaValue.String "", false
          JinjaValue.String "x", true
          JinjaValue.List [], false
          JinjaValue.List [JinjaValue.None], true
          JinjaValue.Object Map.empty, false
          JinjaValue.Object (Map.ofList ["x", JinjaValue.None]), true ]
    for value, expected in cases do
        render (Map.ofList ["value", value]) "{% if value %}yes{% else %}no{% endif %}"
        |> assertText (if expected then "yes" else "no")

[<Fact>]
let ``conditions short circuit and select only the first matching branch`` () =
    render Map.empty "{{ false and missing }}|{{ 'fallback' or missing }}|{{ not 0 }}"
    |> assertText "False|fallback|True"
    render Map.empty "{% if false %}{{ missing }}{% elif true %}chosen{% elif missing %}bad{% else %}bad{% endif %}"
    |> assertText "chosen"
    render Map.empty "{{ 3 < 2 < missing }}" |> assertText "False"
    render Map.empty "{% if false %}ignored{% endif %}" |> assertText ""

[<Fact>]
let ``missing variables and members are strict errors`` () =
    for source in ["{{ missing }}"; "{% if missing %}x{% endif %}"; "{{ true and missing }}"; "{{ false or missing }}"] do
        Assert.Equal(Error (JinjaRuntimeError.MissingVariable ["missing"]), render Map.empty source)
    let context = Map.ofList ["user", JinjaValue.Object Map.empty]
    Assert.Equal(Error (JinjaRuntimeError.MissingVariable ["user"; "name"]), render context "{{ user.name }}")

[<Fact>]
let ``loops expose iteration metadata and restore shadowed scopes`` () =
    let context =
        Map.ofList
            [ "item", JinjaValue.String "outer"
              "loop", JinjaValue.String "original"
              "items", JinjaValue.List [JinjaValue.String "A"; JinjaValue.String "B"] ]
    render context "{% for item in items %}{{ loop.index }}/{{ loop.index0 }}/{{ loop.length }}/{{ loop.first }}/{{ loop.last }}:{{ item }};{% for item in items %}{{ item }}{% endfor %}{{ item }};{% endfor %}{{ item }}:{{ loop }}"
    |> assertText "1/0/2/True/False:A;ABA;2/1/2/False/True:B;ABB;outer:original"
    render (Map.ofList ["items", JinjaValue.List []]) "{% for x in items %}{{ missing }}{% endfor %}"
    |> assertText ""
    Assert.Equal(JinjaValue.String "outer", context["item"])

[<Fact>]
let ``comparisons support numeric booleans structural values and membership`` () =
    let context =
        Map.ofList
            [ "items", JinjaValue.List [JinjaValue.Integer 1L; JinjaValue.String "x"]
              "same", JinjaValue.List [JinjaValue.Boolean true; JinjaValue.String "x"]
              "object", JinjaValue.Object (Map.ofList ["role", JinjaValue.String "user"]) ]
    render context "{{ 0 < 1 <= 2 }}|{{ 2 > 1 >= 1 }}|{{ true == 1 }}|{{ false != 0 }}|{{ items == same }}|{{ 'role' in object }}|{{ 'x' in items }}|{{ 'bc' in 'abcd' }}|{{ 'z' not in items }}"
    |> assertText "True|True|True|False|True|True|True|True|True"
    render context "{{ object == object }}|{{ 1 == '1' }}|{{ none == none }}|{{ 'a' < 'b' }}"
    |> assertText "True|False|True|True"

[<Theory>]
[<InlineData("{{ 'x'.name }}")>]
[<InlineData("{{ missing | upper }}")>]
let ``unsupported syntax is rejected before evaluation`` source =
    Assert.True(JinjaParser.parse source |> Result.isError)

[<Fact>]
let ``invalid operations return errors instead of coercing values`` () =
    let context = Map.ofList ["items", JinjaValue.List []; "obj", JinjaValue.Object Map.empty; "x", JinjaValue.String "x"]
    for source in ["{{ items }}"; "{{ obj }}"; "{{ x.name }}"; "{{ 1 < '2' }}"; "{{ 1 in 'abc' }}"; "{{ 1 in obj }}"; "{{ 1 in 2 }}"; "{% for x in 2 %}{% endfor %}"; "{% for loop in items %}{% endfor %}"] do
        match render context source with
        | Error (JinjaRuntimeError.InvalidOperation _) -> ()
        | actual -> failwithf "Expected operation error for %s: %A" source actual

[<Fact>]
let ``output is culture invariant`` () =
    let previous = CultureInfo.CurrentCulture
    try
        CultureInfo.CurrentCulture <- CultureInfo.GetCultureInfo "fr-FR"
        render Map.empty "{{ -123 }}|{{ true }}|{{ false }}|{{ none }}" |> assertText "-123|True|False|None"
    finally CultureInfo.CurrentCulture <- previous

[<Fact>]
let ``budgets bound output work and caller constructed nesting`` () =
    let limits = { JinjaRenderLimits.Default with MaxOutputLength = 3; MaxSteps = 20; MaxDepth = 4 }
    JinjaRuntime.renderWithLimits limits Map.empty (parse "abc") |> assertText "abc"
    let nested = [JinjaNode.Output (List.fold (fun inner _ -> JinjaExpression.Not inner) (JinjaExpression.Literal (JinjaLiteral.Boolean true)) [1..6])]
    let items = Map.ofList ["items", JinjaValue.List (List.replicate 30 JinjaValue.None)]
    let cases =
        [ Map.empty, parse "abcd"
          Map.empty, parse "ab{{ 'cd' }}"
          items, parse "{% for item in items %}{% endfor %}"
          Map.empty, nested ]
    for context, ast in cases do
        match JinjaRuntime.renderWithLimits limits context ast with
        | Error (JinjaRuntimeError.LimitExceeded _) -> ()
        | actual -> failwithf "Expected limit error: %A" actual
    JinjaRuntime.renderWithLimits limits Map.empty (parse "ok") |> assertText "ok"
    Assert.Throws<ArgumentException>(fun () -> JinjaRuntime.renderWithLimits { limits with MaxDepth = 129 } Map.empty [] |> ignore) |> ignore
    Assert.Throws<ArgumentNullException>(fun () -> JinjaRuntime.render Unchecked.defaultof<_> [] |> ignore) |> ignore

[<Fact>]
let ``one AST can render independent contexts repeatedly`` () =
    let ast = parse "{{ value }}"
    let results =
        [| 1..20 |] |> Array.Parallel.map (fun value ->
            JinjaRuntime.render (Map.ofList ["value", JinjaValue.Integer (int64 value)]) ast)
    results |> Array.iteri (fun index actual -> assertText (string (index + 1)) actual)
