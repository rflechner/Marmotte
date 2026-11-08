module Marmotte.Core.ParserCombinator

open System
open System.Collections.Generic
open Helpers

type ParsingContext =
    { Remaining : char ReadOnlyMemory
      Position : Position }
    static member FromString (s:string) =
        { Remaining = s.AsMemory()
          Position = Position.Zero }
    member ctx.Forward n =
        { Remaining = ctx.Remaining.Slice n
          Position = ctx.Position.ForwardMemory (ctx.Remaining.Slice(0, n)) }
    member ctx.ForwardChar c =
        { Remaining = ctx.Remaining.Slice 1
          Position = ctx.Position.IncFromChar c }
    member ctx.ForwardMemory (m : char ReadOnlyMemory) =
        { Remaining = ctx.Remaining.Slice m.Length
          Position = ctx.Position.ForwardMemory m }
and Position =
    { Line : int
      Column : int
      Offset : int }
    static member Zero
        with get () =
            { Line = 0; Column = 0; Offset = 0; }
    member previous.IncFromChar c =
        { Offset = previous.Offset + 1
          Line = if c = '\n' then previous.Line + 1 else previous.Line
          Column = if c = '\n' then 0 else previous.Column + 1 }
    member previous.ForwardMemory (m : char ReadOnlyMemory) =
        let mutable line = previous.Line
        let mutable column = previous.Column
        let span = m.Span
        for i = 0 to m.Length-1 do
            let c = span.Item i
            if c = '\n'
            then
                line <- line + 1
                column <- 0
            else
                column <- column + 1
        { Offset = previous.Offset + m.Length
          Line = line
          Column = column }

type ReadResult<'t> =
    | Failure of string * ParsingContext
    | Success of 't * ParsingContext

[<RequireQualifiedAccess>]
module ReadResult =
    let map (f : 't -> 'u) (r : 't ReadResult) : 'u ReadResult =
        match r with
        | Failure (e, ctx) -> Failure (e, ctx)
        | Success (v, ctx) -> Success (f v, ctx)

type Parser<'t> = ParsingContext -> ReadResult<'t>

let run (parser:'t Parser) input = 
  parser input

let andThen (a:'a Parser) (b:'b Parser) : ('a * 'b) Parser =
  let f ctx =
    match run a ctx with
    | Failure (error, position) -> Failure (error, position)
    | Success (v1, ctx) ->
      match run b ctx with
      | Failure (error, p) -> Failure (error, p)
      | Success (v2,ctx) ->
          Success ((v1,v2),ctx)
  f

let (.>>.) = andThen

let orElse parser1 parser2 =
  let f input =
    let result1 = run parser1 input
    match result1 with
    | Success _ -> result1
    | Failure _ -> 
        let result2 = run parser2 input
        result2
  f

let ( <|> ) = orElse

let choice (listOfParsers : 't Parser list) : 't Parser = 
  List.reduce ( <|> ) listOfParsers
  
let pChar c : char Parser =
    fun (ctx: ParsingContext) ->
        let span = ctx.Remaining.Span
        if span.Length < 1
        then Failure ("end of buffer", ctx)
        else
            if span.Item 0 = c
            then Success (c, ctx.ForwardChar c)
            else Failure ("not expected char", ctx)

let anyOf listOfChars = 
  listOfChars
  |> List.map pChar
  |> choice

let satisfy (f:char->bool) =
  fun (ctx: ParsingContext) ->
    if ctx.Remaining.Length <= 0
    then Failure ("End of string", ctx)
    else
      let c = ctx.Remaining.Span.Item 0
      if f c
      then Success (c, ctx.ForwardChar c)
      else
        let msg = $"'%c{c}' does not satisfy condition"
        Failure (msg, ctx)

let manySatisfy (f:char->bool) =
  fun (ctx: ParsingContext) ->
    let span = ctx.Remaining.Span
    if span.Length <= 0
    then 
      Failure ("End of string", ctx)
    else
      let mutable i = 0
      while i < span.Length && span.Item i |> f do
        i <- i + 1
      if i > 0
      then Success (ctx.Remaining.Slice(0, i), ctx.Forward i)
      else Failure ("Not matching chars", ctx)

let pDigit = satisfy Char.IsDigit
let pLetter = satisfy Char.IsLetter
let pLetterOrDigit = satisfy Char.IsLetterOrDigit
let pWhiteSpace = satisfy Char.IsWhiteSpace

let pString (str:string) =
  fun (ctx: ParsingContext) ->
    let span = ctx.Remaining.Span
    let pattern = str.AsSpan()
    if span.StartsWith pattern
    then Success (str, ctx.Forward pattern.Length)
    else Failure ("Expected string not found", ctx)

let pMemory (pattern:char ReadOnlyMemory) =
  fun (ctx: ParsingContext) ->
    let span = ctx.Remaining.Span
    if span.StartsWith(pattern.Span)
    then Success (ctx.Remaining.Slice(0, pattern.Length), ctx.Forward pattern.Length)
    else Failure ("Expected pattern not found", ctx)

let pStringIgnoreCase (pattern:char ReadOnlyMemory) =
  fun (ctx: ParsingContext) ->
    let span = ctx.Remaining.Span
    if span.StartsWith(pattern.Span, StringComparison.OrdinalIgnoreCase)
    then Success (ctx.Remaining.Slice(0, pattern.Length), ctx.Forward pattern.Length)
    else Failure ("Expected pattern not found", ctx)

let many (p : 't Parser) =
  fun (ctx: ParsingContext) ->
    let rec loop str (acc:Queue<'t>) =
      match run p str with
      | Success (v1, rest) ->
          acc |> Queue.enqueue v1 |> loop rest
      | Failure (err, ctx) -> 
          if acc |> Queue.isEmpty
          then Failure (err, ctx)
          else Success (acc |> Seq.toList, str)
    let acc = Queue<'t>()
    loop ctx acc

let manyMem (p : Parser<'a>) : ParsingContext -> char ReadOnlyMemory ReadResult =
  fun (origin: ParsingContext) ->
    let rec loop str acc =
      match run p str with
      | Success (_, rest) ->
          loop rest (acc + 1)
      | Failure (err, ctx) -> 
          if acc <= 0
          then Failure (err, ctx)
          else Success (origin.Remaining.Slice(0, acc), str)
    loop origin 0

let manyChars (p : char Parser) =
  fun (ctx: ParsingContext) ->
    match run (manyMem p) ctx with
    | Success (v, rest) -> Success (v, rest)
    | Failure (err, ctx) -> Failure (err, ctx)


let manyWithCount (fc:int->bool) p =
  fun (ctx: ParsingContext) ->
    let rec loop str acc =
      match run p str with
      | Success (v1, rest) ->
          loop rest (v1 --> acc)
      | Failure (err,ctx) -> 
          if fc acc.Length
          then Success (acc, str)
          else Failure (err,ctx)
    loop ctx []

let manyCharsWithCount fc (p:Parser<char>) =
  fun (ctx: ParsingContext) ->
    match run (manyWithCount fc p) ctx with
    | Success (v, rest) ->
        let value = v |> createString
        Success (value, rest)
    | Failure (err,ctx) -> Failure (err,ctx)

let between left right middle =
  fun (origin: ParsingContext) ->
    match run left origin with
    | Failure (err, ctx) -> Failure (err, ctx)
    | Success (v1, rest1) ->
        match run middle rest1 with
        | Failure (err, _) -> Failure (err, origin)
        | Success (v2, rest2) ->
            match run right rest2 with
            | Failure (err, _) -> Failure (err, origin)
            | Success (v3, rest3) ->
                Success ((v1,v2,v3), rest3)

let sepBy (p : 'a Parser) (sep : 'b Parser) : ParsingContext -> 'a list ReadResult =
  fun (origin: ParsingContext) ->
    let rec loop input (acc : 'a Queue) =
      match run p input with
      | Failure (err, _) -> Failure (err, origin)
      | Success (v1, rest1) ->
          match run sep rest1 with
          | Failure _ when acc |> Queue.isEmpty |> not ->
              Success (acc |> Queue.enqueue v1 |> Seq.toList, rest1)
          | Failure (err, _) -> Failure (err, origin)
          | Success (_, rest2) ->
              match acc |> Queue.enqueue v1 |> loop rest2 with
              | Failure _ when acc |> Queue.isEmpty |> not ->
                  Success (acc |> Seq.toList, rest1)
              | other -> other
    loop origin (Queue<'a>())

let map (f:'t -> 'u) (parser:'t Parser) : 'u Parser =
  fun (ctx: ParsingContext) ->
    match run parser ctx with
    | Failure (err, ctx) -> Failure (err, ctx)
    | Success (v1, rest) -> Success (f v1, rest)

let (||>) p f = map f p

let (>>.) a b =
  fun (ctx: ParsingContext) ->
    match run (a .>>. b) ctx with
    | Success ((_,v2), rest) -> Success (v2, rest)
    | Failure (err, ctx) -> Failure (err, ctx)

let (.>>) a b =
  fun (ctx: ParsingContext) ->
    match run (a .>>. b) ctx with
    | Success ((v1,_), rest) -> Success (v1, rest)
    | Failure (err, ctx) -> Failure (err, ctx)

let peol = 
  (pChar '\r' >>. pChar '\n')
    <|> (pChar '\n' >>. pChar '\r')
    <|> pChar '\r' 
    <|> pChar '\n'
  ||> fun _ -> ()

let pUntilStr (s:string) : char ReadOnlyMemory Parser =
  fun (ctx: ParsingContext) ->
    let input = s.AsSpan()
    match ctx.Remaining.Span.IndexOf input with
    | -1 -> Failure ((sprintf "'%s' not found" s), ctx)
    | i ->
        let m = ctx.Remaining
        let text = m.Slice (0, i)
        let rest = m.Slice text.Length
        let ctx = { ctx with Remaining=rest; Position=ctx.Position.ForwardMemory(text) }
        Success(text, ctx)

let pReadTo (text:char ReadOnlyMemory) : char ReadOnlyMemory Parser =
    fun (ctx: ParsingContext) ->
        let span = ctx.Remaining.Span
        match span.IndexOf text.Span with
        | i when i <= 0 -> Failure ("nothing matched", ctx)
        | i ->
            let left = ctx.Remaining.Slice(0, i)
            let rest = ctx.Remaining.Slice(i)
            let ctx = { ctx with Remaining=rest; Position=ctx.Position.ForwardMemory left }
            Success(left, ctx)

let pUntil (p : 't Parser) : char ReadOnlyMemory Parser =
  fun (origin: ParsingContext) ->
    let rec loop ctx (acc:int) =
      match run p ctx with
      | Success _ ->
          if acc <= 0
          then Failure ("nothing matched", origin)
          else
              let r = origin.Remaining.Slice(0, acc)
              Success (r, ctx)
      | Failure _ ->
          if ctx.Remaining.Length <= 1
          then Failure ("nothing matched", origin)
          else
            let rest = (ctx.Forward 1)
            loop rest (acc + 1)
    loop origin 0

let pSkip p =
  fun (ctx: ParsingContext) ->
    let rec loop ctx acc =
      match run p ctx with
      | Success (_, rest) ->
          loop rest (acc+1)
      | Failure _ -> Success ((), ctx)
    loop ctx 0

let zeroOrMany p =
  fun (ctx: ParsingContext) ->
    match run (manyMem p) ctx with
    | Success (v, r) -> Success (v, r)
    | Failure _ -> Success (ReadOnlyMemory.Empty, ctx)

let zeroOrOne p =
  fun (ctx: ParsingContext) ->
    match run p ctx with
    | Success (v, r) -> Success (Some v, r)
    | Failure _ -> Success (None, ctx)

let pOptional (parser:'t Parser) =
    fun (ctx: ParsingContext) ->
      match run parser ctx with
      | Success (v, rest) -> Success (Some v, rest)
      | Failure (_, ctx) -> Success (None, ctx)

let notSatisfy (f:char->bool) : char Parser =
    satisfy (f >> not)

let space = pChar ' ' <|> pChar '\t'
let pNewLine = pChar '\n' <|> pChar '\r'
let spaces = zeroOrMany space
let oneOrManySpaces = space .>>. spaces
let skipSpaces = pSkip space
let spaceOrNewLine = space <|> pNewLine
let skipSpacesAndNewLines = pSkip spaceOrNewLine

let anyCharOf (chars : char Set) =
  satisfy chars.Contains

let manyCharsIn chars =
  manyChars (anyCharOf chars)

let manyCharsNotIn (chars : char Set) =
  manyChars (notSatisfy chars.Contains)
