/// Synchronous, bounded evaluation of the supported Jinja syntax using explicit data.
module Marmotte.Core.JinjaRuntime

open System
open System.Globalization
open System.Text

// A private exception unwinds nested evaluation without publishing partial output.
// Only this exception is converted to the public Result contract.
exception private EvaluationFailure of JinjaRuntimeError

let private fail error = raise (EvaluationFailure error)
let private invalid message = fail (JinjaRuntimeError.InvalidOperation message)

let private truthy = function
    | JinjaValue.None -> false
    | JinjaValue.Boolean value -> value
    | JinjaValue.Integer value -> value <> 0L
    | JinjaValue.String value -> value.Length > 0
    | JinjaValue.List values -> not (List.isEmpty values)
    | JinjaValue.Object values -> not (Map.isEmpty values)

let private numeric = function
    | JinjaValue.Integer value -> Some value
    | JinjaValue.Boolean value -> Some (if value then 1L else 0L)
    | _ -> None

let private scalarText = function
    | JinjaValue.String value -> value
    | JinjaValue.Integer value -> value.ToString(CultureInfo.InvariantCulture)
    | JinjaValue.Boolean true -> "True"
    | JinjaValue.Boolean false -> "False"
    | JinjaValue.None -> "None"
    | JinjaValue.List _ | JinjaValue.Object _ -> invalid "Only scalar values can be written to output."

/// Render an AST with immutable caller data and explicit resource limits.
/// Missing values are strict errors, including in conditions, unless short-circuiting
/// avoids evaluation. No HTML escaping, whitespace trimming, implicit token insertion,
/// reflection, I/O or re-parsing of context values occurs. Loops accept lists only.
/// Returns Error without partial output. Null top-level arguments raise ArgumentNullException;
/// invalid limits raise ArgumentException. Nested .NET nulls are not valid JinjaValue data.
/// See docs/chat-prompts.md for truthiness, comparison and scope semantics.
let renderWithLimits (limits: JinjaRenderLimits) (context: Map<string, JinjaValue>) (nodes: JinjaNode list)
    : Result<string, JinjaRuntimeError> =
    ArgumentNullException.ThrowIfNull limits
    ArgumentNullException.ThrowIfNull context
    ArgumentNullException.ThrowIfNull nodes
    if limits.MaxOutputLength <= 0 || limits.MaxSteps <= 0 || limits.MaxDepth <= 0 || limits.MaxDepth > 128 then
        invalidArg "limits" "Output and step limits must be positive; depth must be between 1 and 128."

    let output = StringBuilder()
    let mutable remainingSteps = limits.MaxSteps

    let step depth =
        if depth > limits.MaxDepth then
            fail (JinjaRuntimeError.LimitExceeded "Evaluation depth exceeded.")
        if remainingSteps = 0 then
            fail (JinjaRuntimeError.LimitExceeded "Evaluation step budget exhausted.")
        remainingSteps <- remainingSteps - 1

    let append (text: string) =
        if text.Length > limits.MaxOutputLength - output.Length then
            fail (JinjaRuntimeError.LimitExceeded "Output length limit exceeded.")
        output.Append text |> ignore

    let lookup scope path =
        let rec memberValue value remaining =
            match remaining with
            | [] -> value
            | name :: rest ->
                step 0
                match value with
                | JinjaValue.Object members ->
                    match Map.tryFind name members with
                    | Some next -> memberValue next rest
                    | None -> fail (JinjaRuntimeError.MissingVariable path)
                | _ -> invalid "Member access requires an object."
        match path with
        | [] -> invalid "A variable path must not be empty."
        | name :: rest ->
            match Map.tryFind name scope with
            | Some value -> memberValue value rest
            | None -> fail (JinjaRuntimeError.MissingVariable path)

    let rec equal depth left right =
        step depth
        match numeric left, numeric right with
        | Some first, Some second -> first = second
        | _ ->
            match left, right with
            | JinjaValue.None, JinjaValue.None -> true
            | JinjaValue.String first, JinjaValue.String second -> String.Equals(first, second, StringComparison.Ordinal)
            | JinjaValue.List first, JinjaValue.List second -> equalLists (depth + 1) first second
            | JinjaValue.Object first, JinjaValue.Object second ->
                first.Count = second.Count &&
                (first |> Map.forall (fun key value ->
                    step depth
                    match Map.tryFind key second with
                    | Some other -> equal (depth + 1) value other
                    | None -> false))
            | _ -> false
    and equalLists depth first second =
        step depth
        match first, second with
        | [], [] -> true
        | left :: leftRest, right :: rightRest ->
            equal depth left right && equalLists depth leftRest rightRest
        | _ -> false

    let order left right =
        match numeric left, numeric right with
        | Some first, Some second -> compare first second
        | _ ->
            match left, right with
            | JinjaValue.String first, JinjaValue.String second -> String.CompareOrdinal(first, second)
            | _ -> invalid "Ordering requires two numbers/booleans or two strings."

    let contains depth item collection =
        match collection with
        | JinjaValue.List values -> values |> List.exists (equal depth item)
        | JinjaValue.Object members ->
            match item with
            | JinjaValue.String key -> Map.containsKey key members
            | _ -> invalid "Object membership requires a string key."
        | JinjaValue.String text ->
            match item with
            | JinjaValue.String substring -> text.Contains(substring, StringComparison.Ordinal)
            | _ -> invalid "String membership requires a string operand."
        | _ -> invalid "Membership requires a list, object or string."

    let compareValues depth operator left right =
        step depth
        match operator with
        | JinjaComparison.Equal -> equal depth left right
        | JinjaComparison.NotEqual -> not (equal depth left right)
        | JinjaComparison.LessThan -> order left right < 0
        | JinjaComparison.LessThanOrEqual -> order left right <= 0
        | JinjaComparison.GreaterThan -> order left right > 0
        | JinjaComparison.GreaterThanOrEqual -> order left right >= 0
        | JinjaComparison.In -> contains depth left right
        | JinjaComparison.NotIn -> not (contains depth left right)

    let rec evaluate depth scope expression =
        step depth
        let next = evaluate (depth + 1) scope
        match expression with
        | JinjaExpression.Variable path -> lookup scope path
        | JinjaExpression.Literal literal ->
            match literal with
            | JinjaLiteral.String value -> JinjaValue.String value
            | JinjaLiteral.Integer value -> JinjaValue.Integer value
            | JinjaLiteral.Boolean value -> JinjaValue.Boolean value
            | JinjaLiteral.None -> JinjaValue.None
        | JinjaExpression.Not operand -> JinjaValue.Boolean (not (truthy (next operand)))
        | JinjaExpression.And (left, right) ->
            let value = next left
            if truthy value then next right else value
        | JinjaExpression.Or (left, right) ->
            let value = next left
            if truthy value then value else next right
        | JinjaExpression.Compare (first, rest) ->
            let rec chain left comparisons =
                match comparisons with
                | [] -> true
                | (operator, operand) :: remaining ->
                    let right = next operand
                    compareValues (depth + 1) operator left right && chain right remaining
            JinjaValue.Boolean (chain (next first) rest)

    let rec renderNodes depth scope body =
        for node in body do
            step depth
            match node with
            | JinjaNode.Text text -> append text
            | JinjaNode.Comment _ -> ()
            | JinjaNode.Output expression -> evaluate (depth + 1) scope expression |> scalarText |> append
            | JinjaNode.If (branches, elseBody) ->
                let rec select remaining =
                    match remaining with
                    | [] -> elseBody |> Option.iter (renderNodes (depth + 1) scope)
                    | branch :: rest ->
                        if evaluate (depth + 1) scope branch.Condition |> truthy then
                            renderNodes (depth + 1) scope branch.Body
                        else select rest
                select branches
            | JinjaNode.For (variable, collection, loopBody) ->
                if variable = "loop" then invalid "The loop variable name is reserved for iteration metadata."
                match evaluate (depth + 1) scope collection with
                | JinjaValue.List values ->
                    // Count against the budget before computing loop.length, including empty bodies.
                    let mutable length = 0L
                    for _ in values do
                        step depth
                        length <- length + 1L
                    let mutable index = 0L
                    for value in values do
                        step depth
                        let loop =
                            Map.ofList
                                [ "index", JinjaValue.Integer (index + 1L)
                                  "index0", JinjaValue.Integer index
                                  "first", JinjaValue.Boolean (index = 0L)
                                  "last", JinjaValue.Boolean (index = length - 1L)
                                  "length", JinjaValue.Integer length ]
                        let inner = scope |> Map.add variable value |> Map.add "loop" (JinjaValue.Object loop)
                        renderNodes (depth + 1) inner loopBody
                        index <- index + 1L
                | _ -> invalid "For loops require a list."

    try
        renderNodes 1 context nodes
        Ok (output.ToString())
    with EvaluationFailure error -> Error error

/// Render using JinjaRenderLimits.Default. Each call owns its output and budgets;
/// the same parsed AST and immutable context can be shared across concurrent renders.
let render context nodes = renderWithLimits JinjaRenderLimits.Default context nodes
