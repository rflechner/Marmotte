namespace Marmotte.Core

open System
open System.Collections
open System.Collections.Generic

type FastStack<'t>(?size:int) as this =
    let [<Literal>] DefaultSize = 2
    
    let mutable storage =
        size |> Option.defaultValue DefaultSize |> Array.zeroCreate |> Memory<'t>
    
    let mutable current = 0
    
    let resizeIfNeeded () =
        if current >= storage.Length
        then
            let size = if storage.Length <= 0 then DefaultSize else storage.Length
            this.Extend size
    
    member _.Extend (size:int) =
        let next = Memory<'t>(Array.zeroCreate (storage.Length + size))
        storage.CopyTo next
        storage <- next
    
    member _.Push (item:'t) =
        resizeIfNeeded ()
        let span = storage.Span
        (span.Item current) <- item
        current <- current + 1
    
    member _.Pop () =
        if current > 0
        then current <- current - 1
    
    member _.TryPeek (item : 't outref) : bool =
        if current = 0
        then false
        else
            item <- storage.Span.Item (current-1)
            true
    
    member _.ToArray() =
        storage.Slice(0, current).ToArray()
    
    member _.Length
        with get () = current
        
    member internal _.toSeq () =
        seq {
            for i=current-1 downto 0 do
                let o = storage.Span.Item i
                yield o
        }
    
    interface IEnumerable<'t> with        
        member __.GetEnumerator() : 't IEnumerator =
            let s = __.toSeq ()
            s.GetEnumerator()
    
    interface IEnumerable with        
        member __.GetEnumerator() : IEnumerator =
            let s = __.toSeq () |> Seq.map unbox
            s.GetEnumerator() :> IEnumerator

[<RequireQualifiedAccess>]
module FastStack =
    
    let push item (stack:'t FastStack) =
        stack.Push item
        stack
    
    let pop (stack:'t FastStack) =
        stack.Pop()
        stack
    
    let createEmpty () = FastStack<'t>(0)

    let toSeq (stack:'t FastStack) =
        stack.toSeq ()

    let extend (size:int) (stack:'t FastStack) =
        stack.Extend size
        stack
    
    let chooseFirst (f : 't -> 'u option) (stack:'t FastStack) =
        stack |> toSeq |> Seq.choose f |> Seq.tryHead
    
    let chooseFirstPair (f : 't * 't -> 'u option) (stack:'t FastStack) =
        stack |> toSeq
        |> Seq.pairwise
        |> Seq.choose f |> Seq.tryHead

    let tryPeek (stack:'t FastStack) =
        let item : 't ref = ref Unchecked.defaultof<'t>
        if stack.TryPeek(item)
        then Some item.Value
        else None
