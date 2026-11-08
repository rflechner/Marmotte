namespace Marmotte.Core

#nowarn "9"

open System
open System.Collections.Generic

module Helpers =
    let (/>) a b = System.IO.Path.Combine(a, b)

    let appendItem item list =
      list @ [item]

    let (-->) = appendItem

    let createString = List.toArray >> String

[<RequireQualifiedAccess>]
module Stack =
    let isEmpty (stack:'t Stack) =
        stack.Count <= 0
    
    let push i (stack:'t Stack) =
        stack.Push i
        stack
        
    let tryPeek (stack:'t Stack) =
        match stack.TryPeek() with
        | true, v -> Some v
        | false, _ -> None

[<RequireQualifiedAccess>]
module Queue =
    let isEmpty (queue:'t Queue) =
        queue.Count <= 0
    
    let enqueue i (queue:'t Queue) =
        queue.Enqueue i
        queue
        
    let tryPeek (queue:'t Queue) =
        match queue.TryPeek() with
        | true, v -> Some v
        | false, _ -> None

module Perf =

    open FSharp.NativeInterop
    
    let inline stackalloc<'a when 'a: unmanaged> size =
        let p = NativePtr.stackalloc<'a> size |> NativePtr.toVoidPtr
        Span<'a>(p, size)
    
    let mergeMemories (memories : 't ReadOnlyMemory list) =
        let len = memories |> List.sumBy (fun m -> m.Length)
        let result : 't Span = stackalloc len
        let mutable p = 0
        for m in memories do
            m.Span.CopyTo(result.Slice(p))
            p <- p + m.Length
        ReadOnlyMemory<'t>(result.ToArray())
    
module NetInterop =
    
    let createList (items:'t seq) =
        items |> Seq.toList
    
    let createFunction (f:Func<'a,'b>) : 'a -> 'b =
        f.Invoke
