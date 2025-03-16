module Helpers

open System

let ucFirst (text: string) =
    if String.IsNullOrWhiteSpace text
    then text
    else
        let firstChar = text.[0]
        let rest = text.[1..].ToLowerInvariant()
        let upperFirstChar = Char.ToUpper(firstChar)
        $"%c{upperFirstChar}%s{rest}"


