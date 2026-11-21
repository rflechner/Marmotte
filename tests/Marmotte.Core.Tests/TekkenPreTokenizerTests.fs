module Marmotte.Core.TekkenPreTokenizerTests

open System
open System.Text
open System.Text.RegularExpressions
open Marmotte.Core
open Xunit

let private assertPieces (input: string) (expected: string array) =
    let actual = TekkenPreTokenizer.split input
    Assert.Equal<string>(expected, actual)
    Assert.Equal(input, String.concat "" actual)
    Assert.All(actual, fun piece -> Assert.NotEmpty piece)

[<Fact>]
let ``sentence boundaries preserve leading spaces and punctuation`` () =
    assertPieces "Hello, how are you ?" [| "Hello"; ","; " how"; " are"; " you"; " ?" |]
    assertPieces "helloWorld HTTPServer ABCdefXYZ" [| "hello"; "World"; " HTTPServer"; " ABCdef"; "XYZ" |]
    assertPieces "I'm WE'RE can't" [| "I"; "'m"; " WE"; "'RE"; " can"; "'t" |]

[<Fact>]
let ``all Unicode number categories are split one scalar at a time`` () =
    assertPieces "12345 ١٢٣ Ⅷ²" [| "1"; "2"; "3"; "4"; "5"; " "; "١"; "٢"; "٣"; " "; "Ⅷ"; "²" |]
    assertPieces "𝟘𝟙" [| "𝟘"; "𝟙" |]

[<Fact>]
let ``whitespace rules reserve a prefix and attach only through the last line break`` () =
    assertPieces "a  b   " [| "a"; " "; " b"; "   " |]
    assertPieces "a\t\tb" [| "a"; "\t"; "\tb" |]
    assertPieces " \r\n  x" [| " \r\n"; " "; " x" |]
    assertPieces " \n \r\n\t " [| " \n \r\n"; "\t " |]
    assertPieces "\u00a0word\u2028next" [| "\u00a0word"; "\u2028next" |]
    assertPieces "!\r\n/abc" [| "!\r\n/"; "abc" |]
    assertPieces " ?!\n/\nX" [| " ?!\n/\n"; "X" |]

[<Fact>]
let ``case categories and combining marks preserve original spellings`` () =
    assertPieces "éÉÉé e\u0301" [| "é"; "ÉÉé"; " e\u0301" |]
    assertPieces "中文ABC中文" [| "中文ABC中文" |]
    assertPieces "中文ABC" [| "中文"; "ABC" |]
    assertPieces "\u0301ABC" [| "\u0301"; "ABC" |]
    assertPieces "ǅuro" [| "ǅuro" |]
    assertPieces "𐐀𐐨𐐀𐐨" [| "𐐀𐐨"; "𐐀𐐨" |]
    assertPieces "👩‍💻 test" [| "👩‍💻"; " test" |]
    assertPieces "😀Hello" [| "😀Hello" |]
    assertPieces "\u0000x" [| "\u0000x" |]

[<Fact>]
let ``empty input is valid but malformed UTF16 is rejected`` () =
    Assert.Empty(TekkenPreTokenizer.split "")
    Assert.Throws<ArgumentNullException>(fun () -> TekkenPreTokenizer.split null |> ignore) |> ignore
    for text in [ string (char 0xD800); string (char 0xDC00); "ok" + string (char 0xD800) + "x" ] do
        Assert.Throws<EncoderFallbackException>(fun () -> TekkenPreTokenizer.split text |> ignore) |> ignore

[<Fact>]
let ``long case and whitespace runs do not require recursive backtracking`` () =
    let upper = String('A', 20000)
    let spaces = String(' ', 20000)
    assertPieces (upper + "b") [| upper + "b" |]
    assertPieces ("文" + upper) [| "文"; upper |]
    assertPieces (spaces + "x") [| spaces.Substring(1); " x" |]

[<Fact>]
let ``scalar scanner agrees with an independent BMP grammar oracle`` () =
    // Category grammar from docs/tekken.md, evaluated independently by a regex engine.
    // This oracle is BMP-only because that engine classifies UTF-16 code units.
    let prefix = @"[^\r\n\p{L}\p{N}]?"
    let upper = @"[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]"
    let lower = @"[\p{Ll}\p{Lm}\p{Lo}\p{M}]"
    let alternatives =
        [ prefix + upper + "*" + lower + "+"
          prefix + upper + "+" + lower + "*"
          @"\p{N}"
          @" ?[^\s\p{L}\p{N}]+[\r\n/]*"
          @"\s*[\r\n]+"
          @"\s+(?!\S)"
          @"\s+" ]
    let oracle = Regex(String.concat "|" alternatives, RegexOptions.CultureInvariant, TimeSpan.FromSeconds 2.0)
    let random = Random 7123
    let alphabet = "aAbBǅʰ中文éÉ\u0301\u0903\u20dd012Ⅷ²١ !?/\r\n\t\u00a0\u2028\u0000'"
    for _ in 1 .. 2000 do
        let input = String(Array.init (random.Next(80)) (fun _ -> alphabet[random.Next alphabet.Length]))
        let expected = oracle.Matches input |> Seq.cast<Match> |> Seq.map (fun item -> item.Value) |> Seq.toArray
        assertPieces input expected
